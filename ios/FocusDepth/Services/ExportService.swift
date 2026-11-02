import Foundation

/// Captures live in Documents/Captures/<scene_id>_<timestamp>/.
/// With UIFileSharingEnabled and LSSupportsOpeningDocumentsInPlace set in Info.plist,
/// they also appear in the Files app ("On My iPhone > FocusDepth").
enum ExportService {
    static var capturesRoot: URL {
        URL.documentsDirectory.appendingPathComponent("Captures", isDirectory: true)
    }

    static func newCaptureFolder(sceneId: String) throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let folder = capturesRoot.appendingPathComponent("\(sceneId)_\(formatter.string(from: Date()))", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    static func listCaptures() -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: capturesRoot, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles]
        )) ?? []
        return urls.filter(\.hasDirectoryPath).sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    /// Zips a capture folder for AirDrop / ShareLink. NSFileCoordinator's `.forUploading`
    /// option produces a zip of a directory without a third-party library.
    static func zip(_ folder: URL) throws -> URL {
        var coordinatorError: NSError?
        var result: Result<URL, Error> = .failure(CocoaError(.fileWriteUnknown))
        NSFileCoordinator().coordinate(readingItemAt: folder, options: .forUploading, error: &coordinatorError) { zipURL in
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("\(folder.lastPathComponent).zip")
            result = Result {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.copyItem(at: zipURL, to: destination)
                return destination
            }
        }
        if let coordinatorError { throw coordinatorError }
        return try result.get()
    }

    /// PNGs the desktop pipeline produced, if the results folder was copied back into
    /// the capture folder as `results/` (e.g. via the Files app).
    static func resultImages(in folder: URL) -> [URL] {
        let results = folder.appendingPathComponent("results", isDirectory: true)
        let urls = (try? FileManager.default.contentsOfDirectory(at: results, includingPropertiesForKeys: nil)) ?? []
        return urls.filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
