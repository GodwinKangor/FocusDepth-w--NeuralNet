import SwiftUI

/// Lists saved captures. Each capture can be shared as a zip, and shows the desktop
/// pipeline's PNGs once a `results/` folder has been copied back into it.
struct ResultsView: View {
    @State private var captures: [URL] = []

    var body: some View {
        List(captures, id: \.self) { folder in
            NavigationLink {
                CaptureDetailView(folder: folder)
            } label: {
                VStack(alignment: .leading) {
                    Text(folder.lastPathComponent)
                    if let manifest = try? CaptureManifest.read(from: folder) {
                        Text([manifest.purposeLabel, manifest.measurementLabel].compactMap { $0 }.joined(separator: " · "))
                            .font(.caption).foregroundStyle(.secondary)
                        Text(manifest.createdAt, format: .dateTime.month().day().hour().minute().second())
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Captures")
        .overlay {
            if captures.isEmpty { ContentUnavailableView("No captures yet", systemImage: "camera") }
        }
        .onAppear { captures = ExportService.listCaptures() }
    }
}

struct CaptureDetailView: View {
    let folder: URL
    @State private var zipURL: URL?
    @State private var error: String?

    private var requestMismatch: Bool {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("capture_request.json")),
              let request = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let purpose = request["capture_purpose"] as? String else { return false }
        guard let saved = try? CaptureManifest.read(from: folder) else { return true }
        if saved.capturePurpose != purpose { return true }
        if purpose != "scene", let text = request["target_distance_cm_text"] as? String,
           let distance = Double(text) {
            return saved.calibrationTargetDistanceM != distance / 100
        }
        return false
    }

    private var frames: [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return urls.filter { $0.pathExtension.lowercased() == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if requestMismatch {
                    Text("Saved metadata differs from the capture request. Share this folder for diagnosis before using it for calibration.")
                        .foregroundStyle(.orange)
                }
                if let manifest = try? CaptureManifest.read(from: folder) {
                    Text("Saved purpose: \(manifest.purposeLabel)").font(.headline)
                    if let distance = manifest.measurementLabel { Text("Target distance: \(distance)") }
                    Text(manifest.createdAt, format: .dateTime.month().day().hour().minute().second())
                    Text("\(manifest.frames.count) saved frames").font(.caption)
                } else {
                    Text("Incomplete capture: no readable manifest").foregroundStyle(.orange)
                }
                if let zipURL {
                    ShareLink(item: zipURL) { Label("Share capture (.zip)", systemImage: "square.and.arrow.up") }
                } else if let error {
                    Text(error).foregroundStyle(.red)
                }

                let results = ExportService.resultImages(in: folder)
                if !results.isEmpty {
                    Text("Results").font(.headline)
                    ForEach(results, id: \.self) { url in
                        VStack(alignment: .leading) {
                            Text(url.deletingPathExtension().lastPathComponent).font(.caption)
                            LocalImage(url: url)
                        }
                    }
                }

                Text("Frames (\(frames.count))").font(.headline)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                    ForEach(frames, id: \.self) { LocalImage(url: $0) }
                }
            }
            .padding()
        }
        .navigationTitle(folder.lastPathComponent)
        .task {
            do { zipURL = try ExportService.zip(folder) } catch { self.error = error.localizedDescription }
        }
    }
}

private struct LocalImage: View {
    let url: URL

    var body: some View {
        if let image = UIImage(contentsOfFile: url.path) {
            Image(uiImage: image).resizable().scaledToFit()
        } else {
            Color.secondary.opacity(0.2).frame(height: 80)
        }
    }
}
