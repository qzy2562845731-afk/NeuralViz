import { useState } from 'react';
import { AppHeader } from './AppHeader';
import { HeroSection } from './HeroSection';
import { FeatureEntryCards } from './FeatureEntryCards';
import { ModelTabNav, MODEL_TABS } from './ModelTabNav';
import { DynamicShowcase } from './DynamicShowcase';
import { FeatureGrid } from './FeatureGrid';
import { WorkflowSection } from './WorkflowSection';
import { CTASection } from './CTASection';
import { ScrollAnimation, ScrollProgressBar } from '../ScrollAnimation';
import { NeuralNetworkBackground } from './NeuralNetworkBackground';
import { VideoBackground } from './VideoBackground';

/* ============================================
   HomePage — DJI 风格官网首页（完整重构版）
   七屏结构：导航 → Hero → 功能入口卡片 → 模型展示主舞台 → 功能 → 流程 → CTA
   ============================================ */

export function HomePage() {
  const [activeTab, setActiveTab] = useState(MODEL_TABS[0].id);

  return (
    <div className="min-h-screen text-foreground antialiased">
      {/* 动态视频背景（视频文件放在 public/videos/background.mp4） */}
      <VideoBackground
        src="/videos/18333010-uhd_3840_2160_25fps.mp4"
        overlayOpacity={0.55}
        fallback={<NeuralNetworkBackground />}
      />

      {/* 滚动进度条 */}
      <ScrollProgressBar />

      {/* 第0层：全局导航栏（fixed定位，单独处理入场动画） */}
      <AppHeader />

      {/* 第1屏：Hero 主视觉 — 保持通透，让视频完整展示 */}
      <ScrollAnimation animation="slide-up" threshold={0.2} duration={800} once={false}>
        <HeroSection />
      </ScrollAnimation>

      {/* 第1.5屏：功能入口卡片化重构 — 核心主入口 + 常用功能 + 帮助引导 + 最近使用 */}
      <div className="relative bg-gradient-to-b from-[#030712]/85 to-[#030712]/90 backdrop-blur-md">
        <FeatureEntryCards />
      </div>

      {/* 第2屏：模型标签导航 + 动态展示主舞台（核心） */}
      <div
        id="showcase-section"
        className="relative border-y border-white/[0.06] bg-gradient-to-b from-[#030712]/90 via-[#030712]/85 to-[#030712]/90 backdrop-blur-md"
      >
        {/* 粘性导航栏不包滚动动画，避免 sticky 离开视口后隐藏 */}
        <ModelTabNav activeTab={activeTab} onTabChange={setActiveTab} />
        <ScrollAnimation animation="fade" threshold={0.1} delay={200} duration={800} once={false}>
          <DynamicShowcase activeTab={activeTab} />
        </ScrollAnimation>
      </div>

      {/* 第3屏：产品功能亮点 */}
      <div className="relative border-b border-white/[0.06] bg-gradient-to-b from-[#030712]/92 via-[#030712]/88 to-[#030712]/92 backdrop-blur-md">
        <ScrollAnimation animation="slide-up" threshold={0.1} delay={150} duration={800} once={false}>
          <FeatureGrid />
        </ScrollAnimation>
      </div>

      {/* 第4屏：使用流程 */}
      <div className="relative border-b border-white/[0.06] bg-gradient-to-b from-[#030712]/88 via-[#030712]/85 to-[#030712]/88 backdrop-blur-md">
        <ScrollAnimation animation="slide-up" threshold={0.1} delay={150} duration={800} once={false}>
          <WorkflowSection />
        </ScrollAnimation>
      </div>

      {/* 第5屏：行动召唤 */}
      <div className="relative bg-gradient-to-b from-[#030712]/90 via-[#030712]/95 to-[#030712] backdrop-blur-md">
        <ScrollAnimation animation="slide-up" threshold={0.1} delay={100} duration={700} once={false}>
          <CTASection />
        </ScrollAnimation>
      </div>

      {/* 页脚 */}
      <ScrollAnimation animation="fade" threshold={0.1} delay={200} duration={600} once={false}>
        <footer className="border-t border-white/[0.04] bg-[#030712] py-8">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 text-xs text-muted-foreground">
            <span>NeuralViz — Neural Network Visualization Engine</span>
            <span className="hidden sm:inline">Built with React + Tailwind CSS + Three.js</span>
          </div>
        </footer>
      </ScrollAnimation>
    </div>
  );
}
