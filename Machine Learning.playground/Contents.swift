import CreateML
import Foundation

let csvFile = Bundle.main.url(forResource: "training", withExtension: "csv")!
let dataTable = try MLDataTable(contentsOf: csvFile)

print(dataTable)

let classifierColumns = ["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ", "gyroRotationX", "gyroRotationY", "gyroRotationZ", "label"]
let classifierTable = dataTable[classifierColumns]

let (classifierEvaluationTable, classifierTrainingTable) = classifierTable.randomSplit(by: 0.20, seed: 5)

let classifier = try MLClassifier(trainingData: classifierTrainingTable, targetColumn: "label")

/// Classifier training accuracy as a percentage
let trainingError = classifier.trainingMetrics.classificationError
let trainingAccuracy = (1.0 - trainingError) * 100

/// Classifier validation accuracy as a percentage
let validationError = classifier.validationMetrics.classificationError
let validationAccuracy = (1.0 - validationError) * 100

/// Evaluate the classifier
let classifierEvaluation = classifier.evaluation(on: classifierEvaluationTable)

/// Classifier evaluation accuracy as a percentage
let evaluationError = classifierEvaluation.classificationError
let evaluationAccuracy = (1.0 - evaluationError) * 100

let classifierMetadata = MLModelMetadata(author: "Mattia Righetti", shortDescription: "Predicts if a user is sitting or standing", version: "1.0")

// 8. Export for use in Core ML
try classifier.write(to: URL(fileURLWithPath: "Users/mattiarighetti/Desktop/ActivityDetection.mlmodel"), metadata: classifierMetadata)
