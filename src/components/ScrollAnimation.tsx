import { useEffect, useRef, useState, useCallback } from 'react';
import { cn } from '../lib/utils';

export interface ScrollAnimationProps {
  className?: string;
  children: React.ReactNode;
  animation?: 'fade' | 'slide-up' | 'slide-down' | 'slide-left' | 'slide-right' | 'scale' | 'flip';
  threshold?: number;
  delay?: number;
  duration?: number;
  once?: boolean;
}

const animationConfig = {
  fade: {
    hidden: { opacity: 0, transform: 'none' },
    visible: { opacity: 1, transform: 'none' },
  },
  'slide-up': {
    hidden: { opacity: 0, transform: 'translateY(40px)' },
    visible: { opacity: 1, transform: 'translateY(0)' },
  },
  'slide-down': {
    hidden: { opacity: 0, transform: 'translateY(-40px)' },
    visible: { opacity: 1, transform: 'translateY(0)' },
  },
  'slide-left': {
    hidden: { opacity: 0, transform: 'translateX(-40px)' },
    visible: { opacity: 1, transform: 'translateX(0)' },
  },
  'slide-right': {
    hidden: { opacity: 0, transform: 'translateX(40px)' },
    visible: { opacity: 1, transform: 'translateX(0)' },
  },
  scale: {
    hidden: { opacity: 0, transform: 'scale(0.92)' },
    visible: { opacity: 1, transform: 'scale(1)' },
  },
  flip: {
    hidden: { opacity: 0, transform: 'rotateY(12deg) translateZ(-30px)' },
    visible: { opacity: 1, transform: 'rotateY(0) translateZ(0)' },
  },
};

export function ScrollAnimation({
  className,
  children,
  animation = 'fade',
  threshold = 0.15,
  delay = 0,
  duration = 700,
  once = true,
}: ScrollAnimationProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (!hasAnimated || !once) {
            if (timerRef.current) clearTimeout(timerRef.current);
            timerRef.current = setTimeout(() => {
              setIsVisible(true);
              if (once) setHasAnimated(true);
            }, delay);
          }
        } else {
          if (!once) {
            if (timerRef.current) clearTimeout(timerRef.current);
            setIsVisible(false);
          }
        }
      },
      { threshold, rootMargin: '0px 0px -60px 0px' }
    );

    const currentRef = ref.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [threshold, delay, once, hasAnimated]);

  const config = animationConfig[animation];
  const state = isVisible ? config.visible : config.hidden;

  return (
    <div
      ref={ref}
      className={cn('will-change-transform', className)}
      style={{
        opacity: state.opacity,
        transform: state.transform,
        transition: `opacity ${duration}ms cubic-bezier(0.22, 1, 0.36, 1), transform ${duration}ms cubic-bezier(0.22, 1, 0.36, 1)`,
        perspective: animation === 'flip' ? '1000px' : undefined,
      }}
    >
      {children}
    </div>
  );
}

export default ScrollAnimation;

/* ---------- 滚动进度条组件 ---------- */
export function ScrollProgressBar() {
  const [scrollY, setScrollY] = useState(0);
  const [windowHeight, setWindowHeight] = useState(0);
  const [documentHeight, setDocumentHeight] = useState(0);

  const updateScroll = useCallback(() => {
    setScrollY(window.scrollY);
    setWindowHeight(window.innerHeight);
    setDocumentHeight(document.documentElement.scrollHeight);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', updateScroll, { passive: true });
    window.addEventListener('resize', updateScroll);
    updateScroll();

    return () => {
      window.removeEventListener('scroll', updateScroll);
      window.removeEventListener('resize', updateScroll);
    };
  }, [updateScroll]);

  const progress = Math.min(
    (scrollY / (documentHeight - windowHeight)) * 100,
    100
  );

  return (
    <div className="fixed top-0 left-0 z-[100] h-0.5 w-full bg-white/[0.1]">
      <div
        className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-amber-500 transition-all duration-300 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

/* ---------- 视差滚动背景组件 ---------- */
export function ParallaxBackground({
  children,
  speed = 0.5,
}: {
  children: React.ReactNode;
  speed?: number;
}) {
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      setOffset(window.scrollY * speed);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [speed]);

  return (
    <div className="relative overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          transform: `translateY(${offset}px)`,
        }}
      >
        {children}
      </div>
    </div>
  );
}
