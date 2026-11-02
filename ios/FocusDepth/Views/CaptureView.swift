import AVFoundation
import SwiftUI

struct CaptureView: View {
    @StateObject private var sweep: FocusSweepController
    @State private var settings = FocusSweepController.Settings()
    @State private var setupError: String?

    init() {
        _sweep = StateObject(wrappedValue: FocusSweepController(camera: CameraSession()))
    }

    // Always go through the StateObject: SwiftUI may re-create this struct at any time.
    private var camera: CameraSession { sweep.camera }

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                CameraPreview(session: camera.session)
                    .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                Form {
                    TextField("Scene ID", text: $settings.sceneId)
                        .textInputAutocapitalization(.never)
                    Stepper("Focus steps: \(settings.steps)", value: $settings.steps, in: 3...24)
                    LabeledContent("Lens min \(settings.minLensPosition, specifier: "%.2f")") {
                        Slider(value: $settings.minLensPosition, in: 0...1)
                    }
                    LabeledContent("Lens max \(settings.maxLensPosition, specifier: "%.2f")") {
                        Slider(value: $settings.maxLensPosition, in: 0...1)
                    }
                    TextField("Notes", text: $settings.notes)
                }
                .frame(maxHeight: 260)

                ProgressView(value: sweep.progress)
                Text(setupError ?? sweep.status).font(.footnote).foregroundStyle(.secondary)

                Button {
                    Task { await sweep.runSweep(settings) }
                } label: {
                    Label("Capture focus sweep", systemImage: "camera.aperture").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(sweep.isCapturing || setupError != nil || settings.minLensPosition >= settings.maxLensPosition)
            }
            .padding()
            .navigationTitle("FocusDepth")
            .toolbar {
                NavigationLink("Captures") { ResultsView() }
            }
            .task {
                do {
                    try await camera.configure()
                    camera.start()
                } catch {
                    setupError = error.localizedDescription
                }
            }
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
