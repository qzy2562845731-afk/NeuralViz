import { useCallback, useState } from 'react';
import type { PlaybackSpeed } from '../types/training';

interface DownloadButtonProps {
  filename?: string;
  currentStep: number;
  maxStep: number;
  speed: PlaybackSpeed;
  isPlaying: boolean;
  isCompleted: boolean;
}

export function DownloadButton({ 
  filename = 'nn-training',
  currentStep,
  maxStep,
  speed,
  isPlaying,
  isCompleted,
}: DownloadButtonProps) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = useCallback(async () => {
    if (isDownloading) return;
    setIsDownloading(true);

    try {
      const appMain = document.querySelector('.app-main') as HTMLElement;
      if (!appMain) {
        console.error('Content not found');
        return;
      }

      const rect = appMain.getBoundingClientRect();
      const scale = 2;

      const canvas = document.createElement('canvas');
      canvas.width = rect.width * scale;
      canvas.height = rect.height * scale;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        console.error('Canvas context not available');
        return;
      }

      ctx.scale(scale, scale);

      ctx.fillStyle = '#0a0a14';
      ctx.fillRect(0, 0, rect.width, rect.height);

      // Header section with training info
      const headerY = 32;
      
      // Title
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 20px Inter, sans-serif';
      ctx.fillText('Neural Network Training Visualization', 32, headerY);
      
      // Training info row
      const infoY = headerY + 36;
      ctx.fillStyle = '#888888';
      ctx.font = '12px Inter, sans-serif';
      
      const statusText = isCompleted ? 'Completed' : isPlaying ? 'Running' : 'Paused';
      const statusColor = isCompleted ? '#4ade80' : isPlaying ? '#fbbf24' : '#6b7280';
      
      const infoItems = [
        { label: 'Step', value: `${currentStep}/${maxStep}` },
        { label: 'Status', value: statusText, color: statusColor },
        { label: 'Speed', value: `${speed}x` },
        { label: 'Date', value: new Date().toLocaleDateString() },
      ];
      
      let infoX = 32;
      infoItems.forEach((item) => {
        ctx.fillStyle = '#666666';
        ctx.fillText(item.label + ':', infoX, infoY);
        ctx.fillStyle = item.color || '#cccccc';
        ctx.fillText(item.value, infoX + 40, infoY);
        infoX += 140;
      });

      // Draw a separator line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(32, infoY + 16);
      ctx.lineTo(rect.width - 32, infoY + 16);
      ctx.stroke();

      // Draw header background to prevent content overlap
      const headerBottom = infoY + 24;
      ctx.fillStyle = 'rgba(10, 10, 20, 0.92)';
      ctx.fillRect(0, 0, rect.width, headerBottom);

      const chartContainers = appMain.querySelectorAll('.echarts-for-react');
      chartContainers.forEach((container) => {
        const canvasEl = container.querySelector('canvas');
        if (canvasEl) {
          const containerRect = container.getBoundingClientRect();
          const relX = containerRect.left - rect.left;
          const relY = containerRect.top - rect.top;
          ctx.drawImage(
            canvasEl,
            relX,
            relY,
            containerRect.width,
            containerRect.height
          );
        }
      });

      // Capture CNN SVG
      const cnnSvg = appMain.querySelector('.cnn-svg') as SVGElement | null;
      if (cnnSvg) {
        const svgRect = cnnSvg.getBoundingClientRect();
        const relX = svgRect.left - rect.left;
        const relY = svgRect.top - rect.top;

        const svgData = new XMLSerializer().serializeToString(cnnSvg);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);

        const img = new Image();
        await new Promise<void>((resolve) => {
          img.onload = () => {
            ctx.drawImage(img, relX, relY, svgRect.width, svgRect.height);
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.onerror = () => {
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.src = svgUrl;
        });
      }

      // Capture heatmaps
      const heatmaps = appMain.querySelectorAll('.heatmap-svg');
      for (const heatmap of Array.from(heatmaps)) {
        const svgEl = heatmap as SVGElement;
        const svgRect = svgEl.getBoundingClientRect();
        const relX = svgRect.left - rect.left;
        const relY = svgRect.top - rect.top;

        const svgData = new XMLSerializer().serializeToString(svgEl);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);

        const img = new Image();
        await new Promise<void>((resolve) => {
          img.onload = () => {
            ctx.drawImage(img, relX, relY, svgRect.width, svgRect.height);
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.onerror = () => {
            URL.revokeObjectURL(svgUrl);
            resolve();
          };
          img.src = svgUrl;
        });
      }

      const dataUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `${filename}-step-${currentStep}-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (error) {
      console.error('Download failed:', error);
    } finally {
      setIsDownloading(false);
    }
  }, [filename, isDownloading, currentStep, maxStep, speed, isPlaying, isCompleted]);

  return (
    <button
      className={`download-btn ${isDownloading ? 'downloading' : ''}`}
      onClick={handleDownload}
      disabled={isDownloading}
      title="Download as PNG"
    >
      {isDownloading ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="spin">
          <circle cx="12" cy="12" r="10" strokeDasharray="32" strokeDashoffset="32" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      )}
      <span className="download-label">Download</span>
    </button>
  );
}
