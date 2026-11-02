import SwiftUI

/// Lists saved captures. Each capture can be shared as a zip, and shows the desktop
/// pipeline's PNGs once a `results/` folder has been copied back into it.
struct ResultsView: View {
    @State private var captures: [URL] = []

    var body: some View {
        List(captures, id: \.self) { folder in
            NavigationLink(folder.lastPathComponent) { CaptureDetailView(folder: folder) }
        }
        .navigationTitle("Captures")
        .overlay {
            if captures.isEmpty { ContentUnavailableView("No captures yet", systemImage: "camera") }
        }
        .onAppear { captures = ExportService.listCaptures() }
    }
}

private struct CaptureDetailView: View {
    let folder: URL
    @State private var zipURL: URL?
    @State private var error: String?

    private var frames: [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return urls.filter { $0.pathExtension.lowercased() == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
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
