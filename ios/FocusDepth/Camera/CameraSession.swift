import AVFoundation

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput
    case noPhotoData
    case permissionDenied

    var errorDescription: String? {
        switch self {
        case .noCamera: "No back wide-angle camera available."
        case .cannotAddInput: "Could not add the camera input."
        case .cannotAddOutput: "Could not add the photo output."
        case .noPhotoData: "The captured photo had no data."
        case .permissionDenied: "Camera access was denied. Enable it in Settings."
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
