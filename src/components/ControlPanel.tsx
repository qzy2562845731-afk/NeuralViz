import type { PlaybackSpeed, TrainingPlayerActions } from '../types/training';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from './ui/card';

interface ControlPanelProps {
  currentStep: number;
  maxStep: number;
  isPlaying: boolean;
  isCompleted: boolean;
  speed: PlaybackSpeed;
  actions: Pick<TrainingPlayerActions, 'togglePlay' | 'stepForward' | 'stepBackward' | 'goToStep' | 'setSpeed' | 'reset'>;
}

const SPEED_OPTIONS: PlaybackSpeed[] = [1, 2, 5];

export function ControlPanel({ currentStep, maxStep, isPlaying, isCompleted, speed, actions }: ControlPanelProps) {
  const { togglePlay, stepForward, stepBackward, goToStep, setSpeed, reset } = actions;
  const progressPercent = maxStep > 0 ? (currentStep / maxStep) * 100 : 0;

  return (
    <Card className="border-border/50">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-sm font-medium tracking-tight">
          Training Controls
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {currentStep}/{maxStep}
          </span>
        </CardTitle>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {/* Play / Navigation */}
        <div className="flex items-center justify-center gap-2">
          <button
            className="ctrl-btn"
            onClick={reset}
            disabled={currentStep === 0 && !isPlaying}
            title="Reset"
          >
            <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="1 4 1 10 7 10" />
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
            </svg>
          </button>

          <button
            className="ctrl-btn"
            onClick={stepBackward}
            disabled={currentStep === 0}
            title="Previous Step"
          >
            <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="19 20 9 12 19 4 19 20" />
              <line x1="5" y1="19" x2="5" y2="5" />
            </svg>
          </button>

          <button
            className={`play-btn ${isPlaying ? 'playing' : ''}`}
            onClick={togglePlay}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg className="size-6" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg className="size-6" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <polygon points="6 3 20 12 6 21 6 3" />
              </svg>
            )}
          </button>

          <button
            className="ctrl-btn"
            onClick={stepForward}
            disabled={isCompleted}
            title="Next Step"
          >
            <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 4 15 12 5 20 5 4" />
              <line x1="19" y1="5" x2="19" y2="19" />
            </svg>
          </button>
        </div>

        {/* Progress slider */}
        <div className="progress-slider">
          <input
            type="range"
            className="absolute inset-0 z-10 h-5 w-full cursor-pointer opacity-0"
            min={0}
            max={maxStep}
            value={currentStep}
            onChange={(e) => goToStep(Number(e.target.value))}
          />
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>

        {/* Speed selector */}
        <div className="speed-selector">
          <span className="text-xs text-muted-foreground">Playback</span>
          <div className="flex gap-1.5">
            {SPEED_OPTIONS.map((s) => (
              <button
                key={s}
                className={`speed-option ${speed === s ? 'active' : ''}`}
                onClick={() => setSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Completion */}
        {isCompleted && (
          <div className="completion-notice">
            <svg className="size-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span>Training Complete</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}