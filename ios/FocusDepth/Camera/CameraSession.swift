import AVFoundation

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput
    case noPhotoData
    case permissionDenied
    case manualFocusUnavailable

    var errorDescription: String? {
        switch self {
        case .noCamera: "No back wide-angle camera available."
        case .cannotAddInput: "Could not add the camera input."
        case .cannotAddOutput: "Could not add the photo output."
        case .noPhotoData: "The captured photo had no data."
        case .permissionDenied: "Camera access was denied. Enable it in Settings."
        case .manualFocusUnavailable: "This camera does not support a manual lens-position sweep."
        }
    }
}

/// Owns the AVCaptureSession. Uses one physical camera (the wide camera) so the
/// system never switches lenses during a sweep.
final class CameraSession: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()
    private(set) var device: AVCaptureDevice?
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

    func configure() async throws {
        guard await Self.requestPermission() else { throw CameraError.permissionDenied }
        try await onQueue {
            guard self.device == nil else { return }  // already configured
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                throw CameraError.noCamera
            }
            self.session.beginConfiguration()
            defer { self.session.commitConfiguration() }
            self.session.sessionPreset = .photo

            let input = try AVCaptureDeviceInput(device: device)
            guard self.session.canAddInput(input) else { throw CameraError.cannotAddInput }
            self.session.addInput(input)
            guard self.session.canAddOutput(self.photoOutput) else { throw CameraError.cannotAddOutput }
            self.session.addOutput(self.photoOutput)
            // Prefer speed so per-frame multi-image fusion (Deep Fusion etc.) is less likely
            // to process each focus step differently.
            self.photoOutput.maxPhotoQualityPrioritization = .speed
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
        let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        settings.flashMode = .off
        settings.photoQualityPrioritization = .speed
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                self.pendingCaptures[settings.uniqueID] = continuation
                self.photoOutput.capturePhoto(with: settings, delegate: self)
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
