import Foundation

/// Written as manifest.json next to the captured frames and read by python/run_pipeline.py.
/// Encoded with `.convertToSnakeCase`, so `lensPositionActual` becomes `lens_position_actual`.
struct CaptureManifest: Codable {
    var sceneId: String
    var device: String
    var camera: String
    var createdAt: Date
    var imageWidth: Int
    var imageHeight: Int
    var whiteBalanceGains: [Float]      // red, green, blue (locked for the whole sweep)
    var geometricDistortionCorrection: Bool
    var frames: [Frame]
    var notes: String

    struct Frame: Codable {
        var file: String
        /// Normalized AVFoundation control value (0 = nearest focus, 1 = farthest). Not meters.
        var lensPositionRequested: Float
        /// `AVCaptureDevice.lensPosition` read back after the focus lock completed.
        var lensPositionActual: Float
        var timestampMs: Int
        var exposureDurationS: Double
        var iso: Float
    }

    func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: directory.appendingPathComponent("manifest.json"))
    }
}
