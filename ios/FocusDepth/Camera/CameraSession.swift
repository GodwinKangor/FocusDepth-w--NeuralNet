import AVFoundation
import CoreImage

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput
    case noPhotoData
    case permissionDenied
    case invalidCalibrationMeasurement
    case savedMetadataMismatch
    case lidarUnavailable
    case lidarFormatUnavailable
    case synchronizedCaptureTimeout
    case missingCalibration
    case exposureLockUnavailable
    case manualFocusUnavailable

    var errorDescription: String? {
        switch self {
        case .noCamera: "No back wide-angle camera available."
        case .cannotAddInput: "Could not add the camera input."
        case .cannotAddOutput: "Could not add the photo output."
        case .noPhotoData: "The captured photo had no data."
        case .permissionDenied: "Camera access was denied. Enable it in Settings."
        case .invalidCalibrationMeasurement: "Enter a positive target distance and a nonnegative measurement uncertainty in centimeters."
        case .savedMetadataMismatch: "Saved capture metadata did not match the requested purpose or distance. Keep this capture for diagnosis."
        case .lidarUnavailable: "No LiDAR camera is available."
        case .lidarFormatUnavailable: "The requested RGB/depth format is unavailable."
        case .synchronizedCaptureTimeout: "No valid synchronized RGB/depth pair arrived within 5 seconds."
        case .missingCalibration: "The depth frame did not include camera calibration."
        case .exposureLockUnavailable: "This camera cannot lock exposure and white balance for the sweep."
        case .manualFocusUnavailable: "This camera does not support a manual lens-position sweep."
        }
    }
}

/// Owns the capture session, using a fixed wide or LiDAR device for each sweep.
final class CameraSession: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()
    private(set) var device: AVCaptureDevice?
    private var lidarMode = false
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let imageContext = CIContext()
    private var pendingPair: (id: UUID, after: CMTime, continuation: CheckedContinuation<SynchronizedPair, Error>)?
    private var droppedVideo = 0
    private var droppedDepth = 0

    struct SynchronizedPair {
        let jpeg: Data
        let depth: Data
        let metadata: Data
        let width: Int
        let height: Int
        let lensPosition: Float
        let exposureSeconds: Double
        let iso: Float
    }

    private let photoOutput = AVCapturePhotoOutput()
    private let queue = DispatchQueue(label: "focusdepth.camera")
    private var pendingCaptures: [Int64: CheckedContinuation<AVCapturePhoto, Error>] = [:]

    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure(lidar: Bool = false) async throws {
        guard await Self.requestPermission() else { throw CameraError.permissionDenied }
        try await onQueue {
            guard self.device == nil || self.lidarMode != lidar else { return }
            if self.session.isRunning { self.session.stopRunning() }
            let type: AVCaptureDevice.DeviceType = lidar ? .builtInLiDARDepthCamera : .builtInWideAngleCamera
            guard let device = AVCaptureDevice.default(type, for: .video, position: .back) else {
                throw lidar ? CameraError.lidarUnavailable : CameraError.noCamera
            }
            self.session.beginConfiguration()
            defer { self.session.commitConfiguration() }
            self.synchronizer?.setDelegate(nil, queue: nil)
            self.synchronizer = nil
            for input in self.session.inputs { self.session.removeInput(input) }
            for output in self.session.outputs { self.session.removeOutput(output) }
            self.device = nil
            self.session.sessionPreset = lidar ? .inputPriority : .photo
            let input = try AVCaptureDeviceInput(device: device)
            guard self.session.canAddInput(input) else { throw CameraError.cannotAddInput }
            self.session.addInput(input)
            if lidar {
                let candidates = device.formats.filter {
                    let size = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                    return size.width == 1920 && size.height == 1440 && !$0.isVideoBinned &&
                        CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                }
                guard let format = candidates.first(where: { format in
                    format.supportedDepthDataFormats.contains {
                        let size = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                        return size.width == 320 && size.height == 240 &&
                            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
                    }
                }), let depthFormat = format.supportedDepthDataFormats.first(where: {
                    let size = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                    return size.width == 320 && size.height == 240 &&
                        CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
                }) else { throw CameraError.lidarFormatUnavailable }
                try device.lockForConfiguration()
                device.activeFormat = format
                device.activeDepthDataFormat = depthFormat
                // Choose a rate supported by both streams rather than inheriting
                // an RGB rate that the depth sensor cannot sustain.
                let rates = format.videoSupportedFrameRateRanges.flatMap { video in
                    depthFormat.videoSupportedFrameRateRanges.compactMap { depth -> Double? in
                        let upper = min(30, video.maxFrameRate, depth.maxFrameRate)
                        return upper >= max(video.minFrameRate, depth.minFrameRate) ? upper : nil
                    }
                }
                guard let fps = rates.max() else {
                    device.unlockForConfiguration()
                    throw CameraError.lidarFormatUnavailable
                }
                let duration = CMTime(seconds: 1 / fps, preferredTimescale: 60000)
                device.activeVideoMinFrameDuration = duration
                device.activeVideoMaxFrameDuration = duration
                device.unlockForConfiguration()
                self.videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.depthOutput.isFilteringEnabled = false
                guard self.session.canAddOutput(self.videoOutput), self.session.canAddOutput(self.depthOutput) else {
                    throw CameraError.cannotAddOutput
                }
                self.session.addOutput(self.videoOutput)
                self.session.addOutput(self.depthOutput)
                // Export both buffers in native sensor orientation, without mirroring.
                for output in [self.videoOutput as AVCaptureOutput, self.depthOutput as AVCaptureOutput] {
                    for connection in output.connections {
                        if connection.isVideoRotationAngleSupported(0) { connection.videoRotationAngle = 0 }
                        if connection.isVideoMirroringSupported {
                            connection.automaticallyAdjustsVideoMirroring = false
                            connection.isVideoMirrored = false
                        }
                    }
                }
                let sync = AVCaptureDataOutputSynchronizer(dataOutputs: [self.videoOutput, self.depthOutput])
                sync.setDelegate(self, queue: self.queue)
                self.synchronizer = sync
            } else {
                guard self.session.canAddOutput(self.photoOutput) else { throw CameraError.cannotAddOutput }
                self.session.addOutput(self.photoOutput)
                self.photoOutput.maxPhotoQualityPrioritization = .speed
            }
            self.lidarMode = lidar
            self.device = device
        }
    }

    func start() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }
    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    /// Advertised capabilities only: does not open additional inputs or prove that
    /// LiDAR, manual focus and multicamera capture work together in one session.
    func capabilityReport() async throws -> Data {
        try await onQueue {
            let discovery = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera, .builtInUltraWideCamera,
                              .builtInTelephotoCamera, .builtInLiDARDepthCamera],
                mediaType: .video, position: .back
            )
            let devices: [[String: Any]] = discovery.devices.map { device in
                let formats: [[String: Any]] = device.formats.map { format in
                    let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                    return [
                        "width": size.width, "height": size.height,
                        "multicam_supported": format.isMultiCamSupported,
                        "video_frame_rates": format.videoSupportedFrameRateRanges.map {
                            ["min_fps": $0.minFrameRate, "max_fps": $0.maxFrameRate]
                        },
                        "depth_formats": format.supportedDepthDataFormats.map { depth -> [String: Any] in
                            let dimensions = CMVideoFormatDescriptionGetDimensions(depth.formatDescription)
                            return ["width": dimensions.width, "height": dimensions.height,
                                    "pixel_format": CMFormatDescriptionGetMediaSubType(depth.formatDescription)]
                        }
                    ]
                }
                return [
                    "id": device.uniqueID, "name": device.localizedName,
                    "type": device.deviceType.rawValue,
                    "selected_for_sweep": device.uniqueID == self.device?.uniqueID,
                    "manual_lens_position_supported": device.isLockingFocusWithCustomLensPositionSupported,
                    "locked_focus_supported": device.isFocusModeSupported(.locked),
                    "custom_exposure_supported": device.isExposureModeSupported(.custom),
                    "locked_white_balance_supported": device.isWhiteBalanceModeSupported(.locked),
                    "formats": formats
                ]
            }
            let report: [String: Any] = [
                "schema_version": 1,
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "multicam_supported": AVCaptureMultiCamSession.isMultiCamSupported,
                "supported_multicam_device_sets": discovery.supportedMultiCamDeviceSets.map {
                    $0.map(\.uniqueID).sorted()
                },
                "devices": devices,
                "validation_scope": "Advertised capabilities only; simultaneous LiDAR/manual-focus operation and accuracy have not been tested."
            ]
            return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        }
    }

    /// Captures one JPEG with flash off and returns the photo once it is processed.
    func capturePhoto() async throws -> AVCapturePhoto {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
                settings.flashMode = .off
                settings.photoQualityPrioritization = .speed
                self.pendingCaptures[settings.uniqueID] = continuation
                self.photoOutput.capturePhoto(with: settings, delegate: self)
            }
        }
    }

    /// Requests the next fresh synchronized pair after the focus settling delay.
    func captureSynchronizedPair() async throws -> SynchronizedPair {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                guard self.lidarMode, self.pendingPair == nil,
                      let clock = self.session.synchronizationClock else {
                    continuation.resume(throwing: CameraError.lidarUnavailable)
                    return
                }
                let id = UUID()
                self.droppedVideo = 0
                self.droppedDepth = 0
                self.pendingPair = (id, CMClockGetTime(clock), continuation)
                self.queue.asyncAfter(deadline: .now() + 5) {
                    guard self.pendingPair?.id == id else { return }
                    self.pendingPair = nil
                    continuation.resume(throwing: CameraError.synchronizedCaptureTimeout)
                }
            }
        }
    }

    private func onQueue<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result { try work() }) }
        }
    }
}

extension CameraSession: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        queue.async {
            guard let continuation = self.pendingCaptures.removeValue(forKey: photo.resolvedSettings.uniqueID) else { return }
            if let error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume(returning: photo)
            }
        }
    }
}


extension CameraSession: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                               didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let pending = pendingPair else { return }
        guard let video = collection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              let depth = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData else { return }
        if video.sampleBufferWasDropped { droppedVideo += 1 }
        if depth.depthDataWasDropped { droppedDepth += 1 }
        guard !video.sampleBufferWasDropped, !depth.depthDataWasDropped,
              CMTimeCompare(video.timestamp, pending.after) >= 0,
              CMTimeCompare(depth.timestamp, pending.after) >= 0,
              let rgb = CMSampleBufferGetImageBuffer(video.sampleBuffer), let device else { return }
        pendingPair = nil
        do {
            let data = depth.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
            guard let calibration = data.cameraCalibrationData else { throw CameraError.missingCalibration }
            let buffer = data.depthDataMap
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw CameraError.noPhotoData }
            var raw = Data()
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            var valid = 0
            for y in 0..<height {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float.self)
                for x in 0..<width {
                    if row[x].isFinite && row[x] > 0 { valid += 1 }
                    var bits = row[x].bitPattern.littleEndian
                    withUnsafeBytes(of: &bits) { raw.append(contentsOf: $0) }
                }
            }
            let image = CIImage(cvPixelBuffer: rgb)
            guard let jpeg = imageContext.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]) else {
                throw CameraError.noPhotoData
            }
            let matrix = calibration.intrinsicMatrix
            let extrinsic = calibration.extrinsicMatrix
            let metadata: [String: Any] = [
                "schema_version": 1,
                "video_timestamp_s": video.timestamp.seconds,
                "depth_timestamp_s": depth.timestamp.seconds,
                "timestamp_delta_s": depth.timestamp.seconds - video.timestamp.seconds,
                "depth_width": width, "depth_height": height,
                "depth_encoding": "float32_little_endian_row_major",
                "depth_units": "meters", "depth_accuracy": data.depthDataAccuracy == .absolute ? "absolute" : "relative",
                "depth_filtered": data.isDepthDataFiltered,
                "valid_depth_fraction": Double(valid) / Double(width * height),
                "invalid_depth_policy": "Preserved; mask nonfinite and nonpositive values on read",
                "dropped_video_while_waiting": droppedVideo,
                "dropped_depth_while_waiting": droppedDepth,
                "orientation": "native_sensor_unmirrored_rotation_0",
                "intrinsics_row_major": (0..<3).map { r in (0..<3).map { c in matrix[c][r] } },
                "intrinsics_reference_width": calibration.intrinsicMatrixReferenceDimensions.width,
                "intrinsics_reference_height": calibration.intrinsicMatrixReferenceDimensions.height,
                "extrinsics_row_major": (0..<3).map { r in (0..<4).map { c in extrinsic[c][r] } },
                "lens_distortion_center": [calibration.lensDistortionCenter.x, calibration.lensDistortionCenter.y],
                "lens_distortion_lookup_base64": calibration.lensDistortionLookupTable?.base64EncodedString() ?? "",
                "inverse_lens_distortion_lookup_base64": calibration.inverseLensDistortionLookupTable?.base64EncodedString() ?? "",
                "lens_position_at_delivery": device.lensPosition,
                "exposure_duration_s_at_delivery": device.exposureDuration.seconds,
                "iso_at_delivery": device.iso
            ]
            pending.continuation.resume(returning: SynchronizedPair(
                jpeg: jpeg, depth: raw,
                metadata: try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]),
                width: CVPixelBufferGetWidth(rgb), height: CVPixelBufferGetHeight(rgb),
                lensPosition: device.lensPosition, exposureSeconds: device.exposureDuration.seconds, iso: device.iso
            ))
        } catch {
            pending.continuation.resume(throwing: error)
        }
    }
}
