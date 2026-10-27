// Apple Vision OCR: prints one JSON line per image with the boxes of text that matches a pattern.
// usage: ocr_boxes <regex> img1.jpg img2.jpg ...   (boxes in pixel coords, origin top-left)
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
let pattern = try! NSRegularExpression(pattern: args[1], options: [.caseInsensitive])
for path in args.dropFirst(2) {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { print("{\"img\":\"\(path)\",\"boxes\":[]}"); continue }
    let W = Double(cg.width), H = Double(cg.height)
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    var boxes: [String] = []
    for obs in (req.results ?? []) {
        guard let cand = obs.topCandidates(1).first else { continue }
        let s = cand.string
        if pattern.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil {
            let b = obs.boundingBox
            let x = b.minX * W, y = (1 - b.maxY) * H, w = b.width * W, h = b.height * H
            boxes.append(String(format: "[%.0f,%.0f,%.0f,%.0f]", x, y, w, h))
            if ProcessInfo.processInfo.environment["OCR_DEBUG"] != nil { FileHandle.standardError.write("\(s) @ \(Int(x)),\(Int(y))\n".data(using: .utf8)!) }
        }
    }
    print("{\"img\":\"\(path)\",\"boxes\":[\(boxes.joined(separator: ","))]}")
}
