import { useEffect, useRef, useState } from 'react';

/* ============================================
   VideoBackground — 全屏视频背景组件
   - 自动播放、静音、循环
   - 视频加载失败时静默降级为神经网络渐变背景
   - 暗化遮罩确保文字可读
   - 响应式 object-cover
   - 性能适配：检测到设备性能不足时，自动切换为静态背景
   - 首帧占位图，加载完成后平滑切入
   - 静默错误处理：不在控制台输出错误信息
   ============================================ */

interface VideoBackgroundProps {
  /** 视频文件路径（相对于 public 目录） */
  src?: string;
  /** 视频格式 MIME 类型 */
  type?: string;
  /** 暗化遮罩透明度 0-1 */
  overlayOpacity?: number;
  /** 加载失败时的降级组件 */
  fallback?: React.ReactNode;
}

/** 神经网络风格的静态渐变背景 */
const StaticFallback = ({ overlayOpacity }: { overlayOpacity: number }) => (
  <div className="absolute inset-0 overflow-hidden">
    <div className="absolute inset-0 bg-gradient-to-b from-[#030712] via-[#0a0c14] to-[#030712]" />
    {/* 神经网络节点效果 */}
    <div className="absolute inset-0 opacity-20">
      <div className="absolute left-1/4 top-1/4 h-64 w-64 rounded-full bg-emerald-500/10 blur-3xl" />
      <div className="absolute right-1/4 top-1/3 h-48 w-48 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="absolute left-1/3 bottom-1/4 h-56 w-56 rounded-full bg-purple-500/10 blur-3xl" />
    </div>
    <div
      className="absolute inset-0"
      style={{
        background: `linear-gradient(to bottom, rgba(3, 7, 18, ${overlayOpacity}) 0%, rgba(3, 7, 18, ${Math.max(0, overlayOpacity - 0.3)}) 40%, rgba(3, 7, 18, ${overlayOpacity}) 100%)`,
      }}
    />
  </div>
);

/** 检测设备性能是否足够播放视频 */
function detectLowPerfDevice(): boolean {
  if (typeof window === 'undefined') return true;
  // 移动设备
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)) {
    return true;
  }
  // 硬件并发数低
  if (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4) {
    return true;
  }
  // 内存不足（仅 Chrome 支持）
  if ((navigator as any).deviceMemory && (navigator as any).deviceMemory < 4) {
    return true;
  }
  // 用户开启了减少动态效果
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return true;
  }
  return false;
}

export function VideoBackground({
  src = '/videos/background.mp4',
  type = 'video/mp4',
  overlayOpacity = 0.65,
  fallback,
}: VideoBackgroundProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [canPlay, setCanPlay] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isLowPerf, setIsLowPerf] = useState(false);
  const errorHandledRef = useRef(false);

  useEffect(() => {
    setIsLowPerf(detectLowPerfDevice());
  }, []);

  useEffect(() => {
    if (isLowPerf) {
      setIsLoading(false);
      setHasError(true);
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    errorHandledRef.current = false;

    const handleCanPlay = () => {
      if (errorHandledRef.current) return;
      setCanPlay(true);
      setIsLoading(false);
    };

    const handleError = () => {
      if (errorHandledRef.current) return;
      errorHandledRef.current = true;
      setHasError(true);
      setIsLoading(false);
      setCanPlay(false);
    };

    const handleLoadedData = () => {
      if (errorHandledRef.current) return;
      setIsLoading(false);
    };

    const handleAbort = () => {
      if (errorHandledRef.current) return;
      errorHandledRef.current = true;
      setHasError(true);
      setIsLoading(false);
    };

    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('error', handleError);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('abort', handleAbort);

    video.play().catch(() => {
      if (!errorHandledRef.current) {
        setIsLoading(false);
      }
    });

    const timeoutId = setTimeout(() => {
      if (!canPlay && !errorHandledRef.current) {
        setHasError(true);
        setIsLoading(false);
      }
    }, 5000);

    return () => {
      clearTimeout(timeoutId);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('error', handleError);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('abort', handleAbort);
    };
  }, [src, isLowPerf]);

  const renderOverlay = () => (
    <>
      <div
        className="absolute inset-0"
        style={{
          background: `linear-gradient(to bottom, rgba(3, 7, 18, ${overlayOpacity}) 0%, rgba(3, 7, 18, ${Math.max(0, overlayOpacity - 0.3)}) 40%, rgba(3, 7, 18, ${overlayOpacity}) 100%)`,
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse at 50% 0%, rgba(3, 7, 18, 0.4) 0%, transparent 60%)',
        }}
      />
    </>
  );

  if (isLowPerf || hasError) {
    return (
      <div className="fixed inset-0 -z-10 overflow-hidden bg-[#030712]">
        {fallback || <StaticFallback overlayOpacity={overlayOpacity} />}
        {renderOverlay()}
      </div>
    );
  }

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-[#030712]">
      {isLoading && <StaticFallback overlayOpacity={overlayOpacity} />}

      <video
        ref={videoRef}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
        className={`absolute left-1/2 top-1/2 h-full w-full min-h-full min-w-full -translate-x-1/2 -translate-y-1/2 object-cover transition-opacity duration-1000 ${
          canPlay ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <source src={src} type={type} />
      </video>

      {renderOverlay()}
    </div>
  );
}

export default VideoBackground;
