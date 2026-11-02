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
    struct Settings {
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
        guard !isCapturing, let device = camera.device else { return }
        isCapturing = true
        progress = 0
        defer { isCapturing = false }
        do {
            let folder = try ExportService.newCaptureFolder(sceneId: settings.sceneId)
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

                let photo = try await camera.capturePhoto()
                guard let data = photo.fileDataRepresentation() else { throw CameraError.noPhotoData }
                let name = String(format: "frame_%03d.jpg", i)
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
                progress = Double(i + 1) / Double(positions.count)
            }

            let manifest = CaptureManifest(
                sceneId: settings.sceneId,
                device: "iPhone 15 Pro",
                camera: "wide",
                createdAt: start,
                imageWidth: dimensions.width,
                imageHeight: dimensions.height,
                whiteBalanceGains: [wbGains.redGain, wbGains.greenGain, wbGains.blueGain],
                geometricDistortionCorrection: device.isGeometricDistortionCorrectionEnabled,
                frames: frames,
                notes: settings.notes
            )
            try manifest.write(to: folder)
            try await restoreAutoModes(device)
            lastCapture = folder
            status = "Saved \(frames.count) frames to \(folder.lastPathComponent)"
        } catch {
            try? await restoreAutoModes(device)
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
