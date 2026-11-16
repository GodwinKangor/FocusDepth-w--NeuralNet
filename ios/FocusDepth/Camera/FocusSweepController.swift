import AVFoundation
import Foundation
import ImageIO

/// Runs a controlled focus sweep: lock exposure and white balance, then for each
/// requested lens position lock focus, wait for the lens to settle, and capture.
///
/// lensPosition is a normalized control value, not a distance. Verify on device
/// that `lensPositionActual` tracks the request and that frames visibly change focus.
@MainActor
final class FocusSweepController: ObservableObject {
    /// Local installation identity: AVFoundation built-in IDs alone do not
    /// distinguish two phones of the same model. Reset on app data removal.
    private static let captureDeviceId: String = {
        let key = "focusdepth.captureDeviceId"
        if let saved = UserDefaults.standard.string(forKey: key) { return saved }
        let value = UUID().uuidString
        UserDefaults.standard.set(value, forKey: key)
        return value
    }()

    struct Settings {
        var capturePurpose = "scene"
        var targetDistanceCm = ""
        var measurementUncertaintyCm = "1"
        var useLiDAR = false
        var sceneId = "scene_01"
        var steps = 10
        var minLensPosition: Float = 0.0
        var maxLensPosition: Float = 1.0
        var settleSeconds: Double = 0.15
        var notes = "Tripod, fixed lighting, static scene"
    }

    @Published var status = "Idle"
    @Published var progress = 0.0
    @Published var isCapturing = false
    @Published var lastCapture: URL?

    let camera: CameraSession

    init(camera: CameraSession) {
        self.camera = camera
    }

    func runSweep(_ settings: Settings) async {
        guard !isCapturing else { return }
        var configuredDevice: AVCaptureDevice?
        var captureFolder: URL?
        isCapturing = true
        lastCapture = nil
        progress = 0
        defer { isCapturing = false }
        do {
            let targetDistance = Double(settings.targetDistanceCm)
            let uncertainty = Double(settings.measurementUncertaintyCm)
            if settings.capturePurpose != "scene" {
                guard let targetDistance, targetDistance.isFinite, targetDistance > 0,
                      let uncertainty, uncertainty.isFinite, uncertainty >= 0 else {
                    throw CameraError.invalidCalibrationMeasurement
                }
            }
            try await camera.configure(lidar: settings.useLiDAR)
            camera.start()
            guard let device = camera.device else { throw CameraError.noCamera }
            configuredDevice = device
            guard device.isLockingFocusWithCustomLensPositionSupported,
                  device.isFocusModeSupported(.locked) else {
                throw CameraError.manualFocusUnavailable
            }
            let folder = try ExportService.newCaptureFolder(sceneId: settings.sceneId)
            captureFolder = folder
            let request: [String: Any] = [
                "capture_purpose": settings.capturePurpose,
                "target_distance_cm_text": settings.targetDistanceCm,
                "measurement_uncertainty_cm_text": settings.measurementUncertaintyCm,
                "use_lidar": settings.useLiDAR,
                "steps": settings.steps,
                "scene_id": settings.sceneId
            ]
            try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys])
                .write(to: folder.appendingPathComponent("capture_request.json"), options: .atomic)
            try Data("Capture in progress; a complete capture has manifest.json.\n".utf8)
                .write(to: folder.appendingPathComponent("capture_status.txt"))
            let capabilities = try await camera.capabilityReport()
            try capabilities.write(to: folder.appendingPathComponent("camera_capabilities.json"), options: .atomic)
            status = "Locking exposure and white balance"
            let wbGains = try await lockExposureAndWhiteBalance(device)

            var frames: [CaptureManifest.Frame] = []
            var dimensions = (width: 0, height: 0)
            let start = Date()
            let positions = Self.lensPositions(settings)
            for (i, position) in positions.enumerated() {
                status = String(format: "Frame %d/%d  lens %.3f", i + 1, positions.count, position)
                try await lockFocus(device, at: position)
                try await Task.sleep(for: .seconds(settings.settleSeconds))
                let actual = device.lensPosition

                let name = String(format: "frame_%03d.jpg", i)
                if settings.useLiDAR {
                    let pair = try await camera.captureSynchronizedPair()
                    let depthName = String(format: "depth_%03d.f32", i)
                    let metadataName = String(format: "pair_%03d.json", i)
                    try pair.jpeg.write(to: folder.appendingPathComponent(name), options: .atomic)
                    try pair.depth.write(to: folder.appendingPathComponent(depthName), options: .atomic)
                    try pair.metadata.write(to: folder.appendingPathComponent(metadataName), options: .atomic)
                    dimensions = (pair.width, pair.height)
                    frames.append(.init(
                        file: name, lensPositionRequested: position, lensPositionActual: pair.lensPosition,
                        timestampMs: Int(Date().timeIntervalSince(start) * 1000),
                        exposureDurationS: pair.exposureSeconds, iso: pair.iso,
                        depthFile: depthName, pairMetadataFile: metadataName
                    ))
                } else {
                    let photo = try await camera.capturePhoto()
                    guard let data = photo.fileDataRepresentation() else { throw CameraError.noPhotoData }
                    try data.write(to: folder.appendingPathComponent(name))
                    let exif = photo.metadata[kCGImagePropertyExifDictionary as String] as? [String: Any]
                    dimensions = (Int(photo.resolvedSettings.photoDimensions.width), Int(photo.resolvedSettings.photoDimensions.height))
                    frames.append(.init(
                        file: name,
                        lensPositionRequested: position,
                        lensPositionActual: actual,
                        timestampMs: Int(Date().timeIntervalSince(start) * 1000),
                        exposureDurationS: (exif?[kCGImagePropertyExifExposureTime as String] as? NSNumber)?.doubleValue ?? device.exposureDuration.seconds,
                        iso: (exif?[kCGImagePropertyExifISOSpeedRatings as String] as? [NSNumber])?.first?.floatValue ?? device.iso
                    ))
                }
                progress = Double(i + 1) / Double(positions.count)
            }

            let manifest = CaptureManifest(
                sceneId: settings.sceneId,
                device: "iPhone 15 Pro",
                camera: settings.useLiDAR ? "lidar_rgb" : "wide",
                createdAt: start,
                imageWidth: dimensions.width,
                imageHeight: dimensions.height,
                whiteBalanceGains: [wbGains.redGain, wbGains.greenGain, wbGains.blueGain],
                geometricDistortionCorrection: device.isGeometricDistortionCorrectionEnabled,
                frames: frames,
                notes: settings.notes,
                capturePurpose: settings.capturePurpose,
                cameraUniqueId: device.uniqueID,
                captureDeviceId: Self.captureDeviceId,
                calibrationTargetDistanceM: settings.capturePurpose == "scene" ? nil : targetDistance.map { $0 / 100 },
                measurementUncertaintyM: settings.capturePurpose == "scene" ? nil : uncertainty.map { $0 / 100 }
            )
            try manifest.write(to: folder)
            let saved = try CaptureManifest.read(from: folder)
            guard saved.capturePurpose == manifest.capturePurpose,
                  saved.calibrationTargetDistanceM == manifest.calibrationTargetDistanceM,
                  saved.measurementUncertaintyM == manifest.measurementUncertaintyM,
                  saved.frames.count == frames.count else {
                throw CameraError.savedMetadataMismatch
            }
            try Data("Complete\n".utf8).write(to: folder.appendingPathComponent("capture_status.txt"))
            try await restoreAutoModes(device)
            camera.start()
            lastCapture = folder
            let measurement = saved.measurementLabel.map { " · \($0)" } ?? ""
            status = "Saved \(saved.purposeLabel)\(measurement): \(frames.count) frames to \(folder.lastPathComponent)"
        } catch {
            if let captureFolder {
                try? Data("Incomplete: \(error.localizedDescription)\n".utf8)
                    .write(to: captureFolder.appendingPathComponent("capture_status.txt"))
            }
            if let configuredDevice { try? await restoreAutoModes(configuredDevice) }
            if camera.device != nil { camera.start() }
            status = "Capture failed: \(error.localizedDescription)"
        }
    }

    static func lensPositions(_ s: Settings) -> [Float] {
        guard s.steps > 1 else { return [s.minLensPosition] }
        return (0..<s.steps).map { i in
            s.minLensPosition + (s.maxLensPosition - s.minLensPosition) * Float(i) / Float(s.steps - 1)
        }
    }

    // MARK: - Device control

    private func lockExposureAndWhiteBalance(_ device: AVCaptureDevice) async throws -> AVCaptureDevice.WhiteBalanceGains {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        guard device.isExposureModeSupported(.custom), device.isWhiteBalanceModeSupported(.locked) else {
            throw CameraError.exposureLockUnavailable
        }
        // Freeze the current auto-exposure result so brightness is constant across the sweep.
        _ = await device.setExposureModeCustom(duration: device.exposureDuration, iso: device.iso)
        let gains = clamped(device.deviceWhiteBalanceGains, max: device.maxWhiteBalanceGain)
        _ = await device.setWhiteBalanceModeLocked(with: gains)
        return gains
    }

    private func lockFocus(_ device: AVCaptureDevice, at position: Float) async throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        // Resumes once the lens has reached the requested position.
        _ = await device.setFocusModeLocked(lensPosition: position)
    }

    private func restoreAutoModes(_ device: AVCaptureDevice) async throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
        if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { device.whiteBalanceMode = .continuousAutoWhiteBalance }
    }

    private func clamped(_ g: AVCaptureDevice.WhiteBalanceGains, max maxGain: Float) -> AVCaptureDevice.WhiteBalanceGains {
        let c = { (v: Float) in Swift.min(Swift.max(v, 1.0), maxGain) }
        return .init(redGain: c(g.redGain), greenGain: c(g.greenGain), blueGain: c(g.blueGain))
    }
}
