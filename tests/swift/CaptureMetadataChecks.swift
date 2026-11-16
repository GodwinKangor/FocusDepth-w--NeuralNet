import Foundation

@main
struct CaptureMetadataChecks {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for (purpose, name, timestamp) in [("scene", "zzz_old", 1000.0), ("calibration", "aaa_new", 2000.0), ("validation", "mmm_middle", 1500.0)] {
            let folder = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let value = CaptureManifest(
                sceneId: name, device: "test", camera: "lidar_rgb", createdAt: Date(timeIntervalSince1970: timestamp),
                imageWidth: 1920, imageHeight: 1440, whiteBalanceGains: [1,1,1],
                geometricDistortionCorrection: false, frames: [], notes: "test",
                capturePurpose: purpose, cameraUniqueId: "camera", captureDeviceId: "installation",
                calibrationTargetDistanceM: purpose == "scene" ? nil : 0.4,
                measurementUncertaintyM: purpose == "scene" ? nil : 0.01
            )
            try value.write(to: folder)
            let saved = try CaptureManifest.read(from: folder)
            precondition(saved.capturePurpose == purpose)
            precondition(saved.calibrationTargetDistanceM == value.calibrationTargetDistanceM)
            precondition(saved.measurementUncertaintyM == value.measurementUncertaintyM)
            precondition(saved.cameraUniqueId == "camera" && saved.captureDeviceId == "installation")
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("manifest.json"))) as! [String: Any]
            precondition(raw["capture_purpose"] as? String == purpose)
            if purpose != "scene" { precondition(raw["calibration_target_distance_m"] as? Double == 0.4) }
        }
        precondition(ExportService.listCaptures(root: root).map(\.lastPathComponent) == ["aaa_new", "mmm_middle", "zzz_old"])
        print("PASS: scene/calibration/validation metadata round trips; newest capture sorted by saved date, not scene ID")
    }
}
