export type LayerName = 'input' | 'conv1' | 'conv2' | 'fc';

export type PlaybackSpeed = 1 | 2 | 5;

export interface LayerActivations {
  input: number[];
  conv1: number[];
  conv2?: number[];
  fc: number[];
}

export interface LayerStats {
  layer: LayerName;
  avg: number;
  max: number;
  min: number;
  sparsity: number;
  variance: number;
}

export interface ActivationStats {
  input: LayerStats;
  conv1: LayerStats;
  conv2?: LayerStats;
  fc: LayerStats;
}

export type FeatureMap = number[][];

export interface FeatureMaps {
  conv1: FeatureMap[];
  conv2?: FeatureMap[];
  fc?: FeatureMap[];
}

export interface TrainingStepData {
  step: number;
  loss: number;
  accuracy: number;
  trainLoss: number;
  valLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
  learningRate: number;
  precision: number;
  recall: number;
  f1Score: number;
  gradientNorm: number;
  weightNorm: number;
  layerActivations: LayerActivations;
  activationStats: ActivationStats;
  featureMaps: FeatureMaps;
  confusionMatrix: number[][];
  predictionDistribution: number[];
}

export interface TrainingPlayerState {
  currentStep: number;
  isPlaying: boolean;
  isCompleted: boolean;
  speed: PlaybackSpeed;
  maxStep: number;
}

export interface TrainingPlayerActions {
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  goToStep: (step: number) => void;
  setSpeed: (speed: PlaybackSpeed) => void;
  reset: () => void;
}

export interface TrainingPlayerOptions {
  data: TrainingStepData[];
  autoPlay?: boolean;
  live?: boolean;
}

export interface TrainingPlayerResult extends TrainingPlayerState, TrainingPlayerActions {
  currentData: TrainingStepData | null;
  visibleData: TrainingStepData[];
}
