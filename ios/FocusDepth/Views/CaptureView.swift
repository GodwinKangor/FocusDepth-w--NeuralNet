import AVFoundation
import SwiftUI

struct CaptureView: View {
    @StateObject private var sweep: FocusSweepController
    @State private var settings = FocusSweepController.Settings()
    @State private var setupError: String?
    @State private var capabilityReportURL: URL?
    @State private var capabilityError: String?

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
                    .overlay {
                        if settings.capturePurpose != "scene" {
                            GeometryReader { geometry in
                                Rectangle().stroke(.yellow, lineWidth: 2)
                                    .frame(width: geometry.size.width * 0.1, height: geometry.size.height * 0.1)
                                    .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                            }
                            .allowsHitTesting(false)
                        }
                    }

                Form {
                    Picker("Capture purpose", selection: $settings.capturePurpose) {
                        Text("Scene").tag("scene")
                        Text("Calibration").tag("calibration")
                        Text("Validation").tag("validation")
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: settings.capturePurpose) { _, purpose in
                        if purpose != "scene" { settings.steps = 24 }
                    }
                    if settings.capturePurpose != "scene" {
                        TextField("Target distance (cm)", text: $settings.targetDistanceCm)
                            .keyboardType(.decimalPad)
                        TextField("Measurement uncertainty (cm)", text: $settings.measurementUncertaintyCm)
                            .keyboardType(.decimalPad)
                        Text("Fill the center with one flat textured target. Measure from the rear camera to its face.")
                            .font(.footnote)
                    }
                    Toggle("Capture LiDAR reference", isOn: $settings.useLiDAR)
                    TextField("Scene ID", text: $settings.sceneId)
                        .textInputAutocapitalization(.never)
                    Stepper("Focus steps: \(settings.steps)", value: $settings.steps, in: 3...40)
                    LabeledContent("Lens min \(settings.minLensPosition, specifier: "%.2f")") {
                        Slider(value: $settings.minLensPosition, in: 0...1)
                    }
                    LabeledContent("Lens max \(settings.maxLensPosition, specifier: "%.2f")") {
                        Slider(value: $settings.maxLensPosition, in: 0...1)
                    }
                    TextField("Notes", text: $settings.notes)
                    if let capabilityReportURL {
                        ShareLink(item: capabilityReportURL) {
                            Label("Share camera capabilities", systemImage: "square.and.arrow.up")
                        }
                        .disabled(sweep.isCapturing)
                    }
                    if let capabilityError {
                        Text(capabilityError).font(.footnote).foregroundStyle(.secondary)
                    }
                }
                .disabled(sweep.isCapturing)
                .frame(maxHeight: 300)

                ProgressView(value: sweep.progress)
                Text(setupError ?? sweep.status).font(.footnote).foregroundStyle(.secondary)

                Button {
                    let requestedSettings = settings
                    Task { await sweep.runSweep(requestedSettings) }
                } label: {
                    Label("Capture \(settings.capturePurpose) sweep", systemImage: "camera.aperture").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(sweep.isCapturing || setupError != nil || settings.minLensPosition >= settings.maxLensPosition)

                if let folder = sweep.lastCapture {
                    NavigationLink("Review latest saved capture") { CaptureDetailView(folder: folder) }
                        .disabled(sweep.isCapturing)
                }
            }
            .padding()
            .navigationTitle("FocusDepth")
            .toolbar {
                NavigationLink("Captures") { ResultsView() }
                    .disabled(sweep.isCapturing)
            }
            .task {
                do {
                    try await camera.configure()
                    camera.start()
                    do {
                        let report = try await camera.capabilityReport()
                        let url = URL.documentsDirectory.appendingPathComponent("camera_capabilities.json")
                        try report.write(to: url, options: .atomic)
                        capabilityReportURL = url
                    } catch {
                        capabilityError = "Could not save camera capabilities: \(error.localizedDescription)"
                    }
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
