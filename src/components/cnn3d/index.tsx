import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from './OrbitControls';
import { export3DModel, downloadScreenshot, type Export3DFormat } from './export3D';
import {
  LAYER_COLORS,
  LAYER_COLORS_HEX,
  DEFAULT_ARCHITECTURE,
  formatLayerShape,
  type NetworkArchitecture,
  type LayerCategory,
  type ViewMode,
} from './types';

/* ============================================
   CNN 3D Viewer — 深度优化版
   - 动态架构加载（不限层数）
   - 类别颜色编码
   - 曲线连接线
   - 分组视觉
   - 多视图模式切换（结构/激活/参数）
   - 轨道相机控制
   - 数据流粒子动画
   - 选中/悬停/激活态三态交互
   ============================================ */

interface CNN3DViewerProps {
  activations?: Record<string, number[]> | null;
  realActivations?: Record<string, number[]>; // 来自推理的真实激活值
  step?: number;
  isPlaying?: boolean;
  speed?: number;
  selectedLayerId?: string | null;
  activeLayerId?: string | null;
  onLayerSelect?: (layerId: string | null) => void;
  onLayerHover?: (layerId: string | null) => void;
  architecture?: NetworkArchitecture;
  viewMode?: ViewMode;
  onViewModeChange?: (mode: ViewMode) => void;
}

/* ---------- 视图模式颜色映射 ---------- */
// 激活视图：冷暖两极，正值越亮（白→青），负值越暗（红→深红）
function getActivationColor(activation: number): THREE.Color {
  const clamped = Math.max(-1, Math.min(1, activation));
  if (clamped > 0) {
    // 正值：暗青 → 亮青 → 白，越正越亮
    const t = clamped;
    return new THREE.Color(0x004d5c).lerp(new THREE.Color(0xcfffff), t);
  } else if (clamped < 0) {
    // 负值：暗红 → 鲜红，越负越暗
    const t = -clamped;
    return new THREE.Color(0x4a0010).lerp(new THREE.Color(0xff2a2a), t);
  }
  return new THREE.Color(0x555555);
}

function getActivationIntensity(activation: number): number {
  const absAct = Math.abs(activation);
  // 高对比发光：暗值微光，亮值强光
  return 0.25 + absAct * 2.2;
}

// 参数视图：热图颜色（冷蓝→暖黄→炽热红），参数越多越亮
function getParameterColor(paramNormalized: number): THREE.Color {
  // paramNormalized: 0 - 1
  if (paramNormalized < 0.25) {
    // 深蓝 → 蓝
    return new THREE.Color(0x1a237e).lerp(new THREE.Color(0x2979ff), paramNormalized * 4);
  } else if (paramNormalized < 0.5) {
    // 蓝 → 青
    return new THREE.Color(0x2979ff).lerp(new THREE.Color(0x00e5ff), (paramNormalized - 0.25) * 4);
  } else if (paramNormalized < 0.75) {
    // 青 → 黄
    return new THREE.Color(0x00e5ff).lerp(new THREE.Color(0xffea00), (paramNormalized - 0.5) * 4);
  } else {
    // 黄 → 橙 → 红
    const t = (paramNormalized - 0.75) * 4;
    return new THREE.Color(0xffea00).lerp(new THREE.Color(0xff3d00), t);
  }
}

function getParameterIntensity(paramNormalized: number): number {
  return 0.4 + paramNormalized * 2.6;
}

// 结构视图：使用中等亮度的基础色
function getStructureIntensity(): number {
  return 0.6;
}

/* ---------- 创建 Canvas Sprite 标签（带 roundRect polyfill） ---------- */
function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  w: number, h: number,
  radius: number
): void {
  const r = Math.min(radius, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function createSpriteLabel(
  text: string,
  subtext?: string,
  accentColor: string = '#7c3aed'
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  if (!ctx) return new THREE.Sprite();

  // 半透明背景
  ctx.fillStyle = 'rgba(12, 14, 23, 0.88)';
  drawRoundRect(ctx, 4, 4, canvas.width - 8, canvas.height - 8, 16);
  ctx.fill();

  // 彩色顶部边
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 3;
  drawRoundRect(ctx, 4, 4, canvas.width - 8, canvas.height - 8, 16);
  ctx.stroke();

  // 主文本
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 42px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 24, 40);

  // 副文本
  if (subtext) {
    ctx.fillStyle = 'rgba(200, 200, 210, 0.85)';
    ctx.font = '24px system-ui, -apple-system, sans-serif';
    ctx.fillText(subtext, 24, 88);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(4.5, 1.2, 1);  // 更大的标签尺寸
  return sprite;
}

/* ---------- 创建曲线连接线（Catmull-Rom） ---------- */
function createCurveConnection(
  startX: number,
  endX: number,
  y: number,
  z: number,
  color: number,
  opacity: number = 0.35
): THREE.Line {
  const points: THREE.Vector3[] = [];
  for (let t = 0; t <= 1; t += 0.05) {
    // 贝塞尔曲线
    const x = THREE.MathUtils.lerp(startX, endX, t);
    const yOff = Math.sin(t * Math.PI) * 0.4;
    points.push(new THREE.Vector3(x, y + yOff, z));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  return new THREE.Line(geometry, material);
}

/* ---------- 主组件 ---------- */
export function CNN3DViewer({
  activations = null,
  realActivations = {},
  step = 0,
  isPlaying = false,
  speed = 1,
  selectedLayerId = null,
  activeLayerId = null,
  onLayerSelect,
  onLayerHover,
  architecture = DEFAULT_ARCHITECTURE,
  viewMode = 'structure',
}: CNN3DViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // 层 mesh 存储
  const layerMeshesRef = useRef<Map<string, THREE.Mesh[]>>(new Map());
  const layerGroupsRef = useRef<Map<string, THREE.Group>>(new Map());
  const layerLabelsRef = useRef<Map<string, THREE.Sprite>>(new Map());
  const connectionLinesRef = useRef<THREE.Line[]>([]);
  const particlesRef = useRef<THREE.Points | null>(null);
  const particlePositionsRef = useRef<Float32Array>(new Float32Array(0));
  const layerPositionsRef = useRef<Map<string, THREE.Vector3>>(new Map());

  const [initStatus, setInitStatus] = useState<'pending' | 'success' | 'error'>('pending');
  const [hoveredLayerName, setHoveredLayerName] = useState<string | null>(null);
  const [legendCollapsed, setLegendCollapsed] = useState(false);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // 支持受控 / 非受控选择模式
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(null);
  // 用 ref 追踪选中层，使动画循环能实时读取最新值
  const selectedLayerRef = useRef<string | null>(null);
  const effectiveSelectedId = selectedLayerId !== undefined ? selectedLayerId : internalSelectedId;
  selectedLayerRef.current = effectiveSelectedId;

  // 用 ref 追踪视图模式，使动画循环能感知当前模式
  const viewModeRef = useRef<ViewMode>(viewMode);
  viewModeRef.current = viewMode;

  // 用 ref 追踪真实激活值，使动画循环能感知
  const realActivationsRef = useRef<Record<string, number[]>>(realActivations);
  realActivationsRef.current = realActivations;

  /* ---------- 计算：每层的 X 位置 ---------- */
  const layers = useMemo(() => architecture.layers, [architecture]);
  const totalLayers = layers.length;
  const spacing = 3.5;
  const startX = -((totalLayers - 1) * spacing) / 2;

  /* ---------- 主初始化 ---------- */
  useEffect(() => {
    setInitStatus('pending');
    let cleanupFn: (() => void) | null = null;

    const initializeScene = () => {
      console.log('[CNN3D] Initializing... containerRef:', containerRef.current);
      if (!containerRef.current || sceneRef.current) {
        console.log('[CNN3D] Skip init - container:', !!containerRef.current, ', scene:', !!sceneRef.current);
        return null;
      }

      const container = containerRef.current;
      const width = container.clientWidth || 800;
      const height = container.clientHeight || 600;
      
      console.log('[CNN3D] Container size:', width, 'x', height);

      try {
        // ---- 场景 ----
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0d1018);
        scene.fog = new THREE.Fog(0x0d1018, 5, 60);
        sceneRef.current = scene;

        // ---- 相机（拉近 15%~20%，让模型更饱满） ----
        const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 500);
        const viewTarget = new THREE.Vector3(0, 0.8, 0);
        camera.position.set(14, 9, 17);
        camera.lookAt(viewTarget);
        cameraRef.current = camera;

        // ---- 渲染器（修复：启用 alpha 确保背景正确） ----
        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x0d1018, 1);
        renderer.shadowMap.enabled = false;
        container.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // ---- 灯光 ----
        const ambient = new THREE.AmbientLight(0xffffff, 0.9);
        scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
        dirLight.position.set(10, 15, 10);
        scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0x7c3aed, 0.6);
        dirLight2.position.set(-8, 8, -5);
        scene.add(dirLight2);

        const pointLight = new THREE.PointLight(0x06b6d4, 0.8, 40);
        pointLight.position.set(0, 5, 8);
        scene.add(pointLight);

        // ---- 网格地面 ----
        const gridHelper = new THREE.GridHelper(50, 50, 0x4c4c6d, 0x2a2a4a);
        gridHelper.position.y = -4;
        (gridHelper.material as THREE.Material).transparent = true;
        (gridHelper.material as THREE.Material).opacity = 0.3;
        scene.add(gridHelper);

        // ---- 创建各层 ----
        console.log('[CNN3D] Creating layers:', layers.length);
        layers.forEach((layer, idx) => {
          const layerX = startX + idx * spacing;
          console.log('[CNN3D] Layer', idx, '-', layer.name, 'at X:', layerX);
          
          const group = new THREE.Group();
          group.position.x = layerX;
          scene.add(group);

          const meshes: THREE.Mesh[] = [];
          const color = LAYER_COLORS[layer.type];
          const hexColor = LAYER_COLORS_HEX[layer.type];

          // 根据类型创建不同的形状
          if (layer.type === 'input') {
            // 输入层：矩形平面（代表图像输入）
            const planeGeo = new THREE.PlaneGeometry(4, 3.5);
            const planeMat = new THREE.MeshStandardMaterial({
              color,
              transparent: true,
              opacity: 0.8,
              metalness: 0.25,
              roughness: 0.55,
              emissive: color,
              emissiveIntensity: 0.45,
              side: THREE.DoubleSide,
            });
            const plane = new THREE.Mesh(planeGeo, planeMat);
            plane.userData = { layerId: layer.id };
            plane.rotation.x = 0.15;
            group.add(plane);
            meshes.push(plane);
          } else if (layer.type === 'conv') {
            // 卷积层：堆叠的小方块矩阵
            const gridDim = 3;
            const cubeSize = 0.5;
            const cubeSpacing = 0.65;

            for (let gx = 0; gx < gridDim; gx++) {
              for (let gy = 0; gy < gridDim; gy++) {
                const geo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
                const mat = new THREE.MeshStandardMaterial({
                  color,
                  transparent: true,
                  opacity: 0.85,
                  metalness: 0.35,
                  roughness: 0.4,
                  emissive: color,
                  emissiveIntensity: 0.5,
                });
                const cube = new THREE.Mesh(geo, mat);
                cube.position.set(
                  (gx - (gridDim - 1) / 2) * cubeSpacing,
                  (gy - (gridDim - 1) / 2) * cubeSpacing,
                  0
                );
                cube.userData = { layerId: layer.id };
                group.add(cube);
                meshes.push(cube);
              }
            }
          } else if (layer.type === 'pool') {
            // 池化层：2x2方块矩阵
            const gridDim = 2;
            const cubeSize = 0.6;
            const cubeSpacing = 0.8;

            for (let gx = 0; gx < gridDim; gx++) {
              for (let gy = 0; gy < gridDim; gy++) {
                const geo = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize * 0.6);
                const mat = new THREE.MeshStandardMaterial({
                  color,
                  transparent: true,
                  opacity: 0.8,
                  metalness: 0.3,
                  roughness: 0.45,
                  emissive: color,
                  emissiveIntensity: 0.45,
                });
                const cube = new THREE.Mesh(geo, mat);
                cube.position.set(
                  (gx - (gridDim - 1) / 2) * cubeSpacing,
                  (gy - (gridDim - 1) / 2) * cubeSpacing,
                  0
                );
                cube.userData = { layerId: layer.id };
                group.add(cube);
                meshes.push(cube);
              }
            }
          } else if (layer.type === 'norm') {
            // 归一化层：圆环形状
            const ringRadius = 1.2;
            const ringThickness = 0.15;
            const ringSegments = 64;
            
            for (let i = 0; i < 3; i++) {
              const ringGeo = new THREE.RingGeometry(
                ringRadius - ringThickness,
                ringRadius + ringThickness,
                ringSegments,
                1,
                0,
                Math.PI * 2
              );
              const mat = new THREE.MeshStandardMaterial({
                color,
                transparent: true,
                opacity: 0.75 - i * 0.15,
                metalness: 0.4,
                roughness: 0.35,
                emissive: color,
                emissiveIntensity: 0.5 - i * 0.1,
                side: THREE.DoubleSide,
              });
              const ring = new THREE.Mesh(ringGeo, mat);
              ring.position.set(0, i * 0.2 - 0.2, 0);
              ring.rotation.x = -Math.PI / 2;
              ring.userData = { layerId: layer.id };
              group.add(ring);
              meshes.push(ring);
            }
          } else if (layer.type === 'dropout') {
            // Dropout层：随机分布的小方块
            const dropCount = 8;
            const blockSize = 0.3;
            
            for (let i = 0; i < dropCount; i++) {
              const geo = new THREE.BoxGeometry(blockSize, blockSize, blockSize);
              const mat = new THREE.MeshStandardMaterial({
                color,
                transparent: true,
                opacity: 0.6 + Math.random() * 0.3,
                metalness: 0.2,
                roughness: 0.6,
                emissive: color,
                emissiveIntensity: 0.3,
              });
              const block = new THREE.Mesh(geo, mat);
              block.position.set(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 0.5
              );
              block.userData = { layerId: layer.id };
              group.add(block);
              meshes.push(block);
            }
          } else if (layer.type === 'fc') {
            // 全连接层：球形节点阵列
            const nodeCount = Math.min(layer.nodeCount, 12);
            const cols = Math.ceil(Math.sqrt(nodeCount));
            const rows = Math.ceil(nodeCount / cols);
            const nodeRadius = 0.32;
            const nodeSpacing = 0.75;

            let placed = 0;
            for (let r = 0; r < rows && placed < nodeCount; r++) {
              for (let c = 0; c < cols && placed < nodeCount; c++) {
                const sphereGeo = new THREE.SphereGeometry(nodeRadius, 20, 20);
                const mat = new THREE.MeshStandardMaterial({
                  color,
                  transparent: true,
                  opacity: 0.9,
                  metalness: 0.5,
                  roughness: 0.3,
                  emissive: color,
                  emissiveIntensity: 0.55,
                });
                const sphere = new THREE.Mesh(sphereGeo, mat);
                sphere.position.set(
                  (c - (cols - 1) / 2) * nodeSpacing,
                  ((rows - 1) / 2 - r) * nodeSpacing,
                  0
                );
                sphere.userData = { layerId: layer.id };
                group.add(sphere);
                meshes.push(sphere);
                placed++;
              }
            }
          } else if (layer.type === 'output') {
            // 输出层：水平条形节点
            const nodeCount = Math.min(layer.nodeCount, 10);
            const nodeRadius = 0.35;
            const nodeSpacing = 0.8;
            for (let i = 0; i < nodeCount; i++) {
              const sphereGeo = new THREE.SphereGeometry(nodeRadius, 20, 20);
              const mat = new THREE.MeshStandardMaterial({
                color: new THREE.Color().setHSL(0, 0.6, 0.5 + i * 0.03),
                transparent: true,
                opacity: 0.9,
                metalness: 0.45,
                roughness: 0.35,
                emissive: new THREE.Color().setHSL(0, 0.7, 0.35),
                emissiveIntensity: 0.45,
              });
              const sphere = new THREE.Mesh(sphereGeo, mat);
              sphere.position.set(
                0,
                ((nodeCount - 1) / 2 - i) * nodeSpacing,
                0
              );
              sphere.userData = { layerId: layer.id };
              group.add(sphere);
              meshes.push(sphere);
            }
          } else {
            // 默认：一个方块
            const geo = new THREE.BoxGeometry(2, 2, 2);
            const mat = new THREE.MeshStandardMaterial({
              color,
              transparent: true,
              opacity: 0.85,
              metalness: 0.35,
              roughness: 0.45,
              emissive: color,
              emissiveIntensity: 0.35,
            });
            const cube = new THREE.Mesh(geo, mat);
            cube.userData = { layerId: layer.id };
            group.add(cube);
            meshes.push(cube);
          }

          layerPositionsRef.current.set(layer.id, new THREE.Vector3(layerX, 0.3, 0));

          const label = createSpriteLabel(
            layer.name,
            `${layer.type.toUpperCase()} · ${formatLayerShape(layer.outputShape)}`,
            hexColor
          );
          label.position.set(0, 2.3, 0);
          group.add(label);
          layerLabelsRef.current.set(layer.id, label);

          // 标签指示线：从层中心垂直向上连接标签底部，增强空间锚定感
          const leaderPoints = [
            new THREE.Vector3(0, 0.4, 0),
            new THREE.Vector3(0, 1.7, 0),
          ];
          const leaderGeometry = new THREE.BufferGeometry().setFromPoints(leaderPoints);
          const leaderMaterial = new THREE.LineBasicMaterial({
            color: hexColor,
            transparent: true,
            opacity: 0.25,
            depthWrite: false,
          });
          const leaderLine = new THREE.Line(leaderGeometry, leaderMaterial);
          group.add(leaderLine);

          layerGroupsRef.current.set(layer.id, group);
          layerMeshesRef.current.set(layer.id, meshes);
        });

        // ---- 创建连接线 ----
        for (let i = 0; i < layers.length - 1; i++) {
          const curStartX = startX + i * spacing;
          const curEndX = startX + (i + 1) * spacing;

          for (let z = -2; z <= 2; z++) {
            const line = createCurveConnection(
              curStartX + 0.8,
              curEndX - 0.8,
              z * 0.15,
              z * 0.2,
              0x667eea,
              0.2
            );
            scene.add(line);
            connectionLinesRef.current.push(line);
          }
        }

        // ---- 数据流粒子系统 ----
        const particleCount = 120;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
          positions[i * 3] = startX;
          positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
          positions[i * 3 + 2] = (Math.random() - 0.5) * 1.2;
          colors[i * 3] = 0.4 + Math.random() * 0.3;
          colors[i * 3 + 1] = 0.85 + Math.random() * 0.15;
          colors[i * 3 + 2] = 0.9;
        }
        particlePositionsRef.current = positions;

        const particleGeometry = new THREE.BufferGeometry();
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const particleMaterial = new THREE.PointsMaterial({
          size: 0.3,
          vertexColors: true,
          transparent: true,
          opacity: 0.8,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        const particles = new THREE.Points(particleGeometry, particleMaterial);
        scene.add(particles);
        particlesRef.current = particles;

        // ---- 轨道相机控制 ----
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.copy(viewTarget);
        controls.minDistance = 6;
        controls.maxDistance = 45;
        controls.minPolarAngle = 0.15;
        controls.maxPolarAngle = Math.PI / 2 - 0.1;
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.autoRotate = false;
        controls.update();
        controlsRef.current = controls;

        // ---- 交互：射线检测 ----
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let hoveredId: string | null = null;

        const onMouseMove = (event: MouseEvent) => {
          const rect = container.getBoundingClientRect();
          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const allMeshes: THREE.Mesh[] = [];
          layerMeshesRef.current.forEach((meshes) => {
            meshes.forEach((m) => allMeshes.push(m));
          });

          const intersects = raycaster.intersectObjects(allMeshes, false);
          const newHovered = intersects.length > 0
            ? (intersects[0].object as THREE.Mesh).userData.layerId as string
            : null;

          if (newHovered !== hoveredId) {
            if (hoveredId && hoveredId !== selectedLayerRef.current && hoveredId !== activeLayerId) {
              const group = layerGroupsRef.current.get(hoveredId);
              if (group) group.scale.setScalar(1);
            }
            hoveredId = newHovered;
            if (hoveredId && hoveredId !== selectedLayerRef.current && hoveredId !== activeLayerId) {
              const group = layerGroupsRef.current.get(hoveredId);
              if (group) group.scale.setScalar(1.1);
            }
            const layerName = hoveredId
              ? layers.find((l) => l.id === hoveredId)?.name ?? null
              : null;
            setHoveredLayerName(layerName);
            if (onLayerHover) onLayerHover(hoveredId);
          }

          container.style.cursor = hoveredId ? 'pointer' : 'grab';
        };

        const onClick = () => {
          if (hoveredId) {
            if (onLayerSelect) {
              onLayerSelect(hoveredId);
            } else {
              setInternalSelectedId(hoveredId === internalSelectedId ? null : hoveredId);
            }
          } else {
            if (onLayerSelect) onLayerSelect(null);
            else setInternalSelectedId(null);
          }
        };

        renderer.domElement.addEventListener('mousemove', onMouseMove);
        renderer.domElement.addEventListener('click', onClick);

        // ---- ResizeObserver ----
        const resizeObserver = new ResizeObserver((entries) => {
          for (const entry of entries) {
            const { width: newWidth, height: newHeight } = entry.contentRect;
            if (newWidth > 0 && newHeight > 0 && cameraRef.current && rendererRef.current) {
              cameraRef.current.aspect = newWidth / newHeight;
              cameraRef.current.updateProjectionMatrix();
              rendererRef.current.setSize(newWidth, newHeight);
            }
          }
        });
        resizeObserver.observe(container);

        // ---- 动画循环 ----
        let animationId = 0;
        let particleTime = 0;
        let pulseTime = 0;

        const animate = () => {
          animationId = requestAnimationFrame(animate);
          
          if (controlsRef.current) {
            controlsRef.current.update();
          }

          pulseTime += 0.04;

          const currentViewMode = viewModeRef.current;

          // 选中层脉冲动画 - 在结构视图中保持原样
          if (selectedLayerRef.current && currentViewMode === 'structure') {
            const selId = selectedLayerRef.current;
            const group = layerGroupsRef.current.get(selId);
            if (group) {
              const pulseScale = 1.15 + Math.sin(pulseTime) * 0.08;
              if (hoveredId !== selId) group.scale.setScalar(pulseScale);
              const meshes = layerMeshesRef.current.get(selId);
              meshes?.forEach((mesh) => {
                if (mesh.material instanceof THREE.MeshStandardMaterial) {
                  mesh.material.emissiveIntensity = 1.8 + Math.sin(pulseTime) * 0.5;
                  mesh.material.opacity = 1;
                }
              });
            }
          }

          // 激活层脉冲动画 - 在结构视图中保持原样
          if (activeLayerId && activeLayerId !== selectedLayerRef.current && currentViewMode === 'structure') {
            const group = layerGroupsRef.current.get(activeLayerId);
            if (group && isPlaying) {
              const pulseScale = 1.08 + Math.sin(pulseTime * 2) * 0.06;
              if (hoveredId !== activeLayerId) group.scale.setScalar(pulseScale);
              const meshes = layerMeshesRef.current.get(activeLayerId);
              meshes?.forEach((mesh) => {
                if (mesh.material instanceof THREE.MeshStandardMaterial) {
                  mesh.material.emissiveIntensity = 1.2 + Math.sin(pulseTime * 2) * 0.3;
                  mesh.material.opacity = 0.95;
                }
              });
            }
          }

          // 非聚焦层降低透明度 - 只在结构视图中限制发光
          if (currentViewMode === 'structure') {
            layerMeshesRef.current.forEach((meshes, id) => {
              if (id !== selectedLayerRef.current && id !== activeLayerId && id !== hoveredId) {
                meshes.forEach((mesh) => {
                  if (mesh.material instanceof THREE.MeshStandardMaterial) {
                    mesh.material.opacity = selectedLayerRef.current || activeLayerId ? 0.55 : 0.85;
                    mesh.material.emissiveIntensity = Math.min(mesh.material.emissiveIntensity, 0.6);
                  }
                });
              }
            });
          }

          // 激活视图 - 持续动态更新激活效果（无需播放也能看到）
          if (currentViewMode === 'activation' && !isPlaying) {
            const layerIds = Array.from(layerMeshesRef.current.keys());
            layerIds.forEach((layerId, layerIdx) => {
              const meshes = layerMeshesRef.current.get(layerId);
              if (!meshes) return;
              
              // 如果有真实激活值，保持静态显示；否则使用动态效果
              const hasRealActivation = realActivationsRef.current && 
                Object.keys(realActivationsRef.current).length > 0;
              
              if (hasRealActivation) {
                // 真实激活值保持静态（已在 useEffect 中设置）
                return;
              }
              
              // 模拟动态效果
              meshes.forEach((mesh, idx) => {
                if (mesh.material instanceof THREE.MeshStandardMaterial) {
                  const activation = Math.sin(pulseTime * 2 + layerIdx * 0.8 + idx * 0.3) * 0.7
                                   + Math.sin(pulseTime * 0.5 + idx) * 0.3;
                  const color = getActivationColor(activation);
                  mesh.material.color.copy(color);
                  mesh.material.emissive.copy(color);
                  mesh.material.emissiveIntensity = getActivationIntensity(activation);
                }
              });
            });
          }

          // 参数视图 - 保持静态效果（已在 useEffect 中应用，不需要每帧更新）

          // 粒子动画
          if (particlesRef.current) {
            if (isPlaying) {
              particleTime += 0.015 * speed;
              const layerPositions = Array.from(layerPositionsRef.current.values());
              const numLayers = layerPositions.length;

              for (let i = 0; i < particlePositionsRef.current.length / 3; i++) {
                const t = ((particleTime + i * 0.015) % 1);
                const totalSeg = numLayers - 1;
                const scaledT = t * totalSeg;
                const segIdx = Math.min(Math.floor(scaledT), totalSeg - 1);
                const localT = scaledT - segIdx;

                if (segIdx < totalSeg) {
                  const startPos = layerPositions[segIdx];
                  const endPos = layerPositions[segIdx + 1];

                  particlePositionsRef.current[i * 3] =
                    THREE.MathUtils.lerp(startPos.x, endPos.x, localT);
                  particlePositionsRef.current[i * 3 + 1] =
                    THREE.MathUtils.lerp(startPos.y, endPos.y, localT) + Math.sin(particleTime * 4 + i) * 0.2;
                  particlePositionsRef.current[i * 3 + 2] =
                    THREE.MathUtils.lerp(startPos.z, endPos.z, localT) + Math.cos(particleTime * 3 + i) * 0.3;
                }
              }
              particlesRef.current.geometry.attributes.position.needsUpdate = true;
            } else {
              particleTime += 0.003;
              for (let i = 0; i < particlePositionsRef.current.length / 3; i++) {
                particlePositionsRef.current[i * 3 + 1] += Math.sin(particleTime * 2 + i) * 0.002;
              }
              particlesRef.current.geometry.attributes.position.needsUpdate = true;
            }
          }

          renderer.render(scene, camera);
        };

        animate();
        setInitStatus('success');
        console.log('[CNN3D] Initialization completed successfully!');

        // ---- 清理 ----
        return () => {
          cancelAnimationFrame(animationId);
          resizeObserver.disconnect();
          renderer.domElement.removeEventListener('mousemove', onMouseMove);
          renderer.domElement.removeEventListener('click', onClick);

          scene.traverse((obj) => {
            if ('geometry' in obj && obj.geometry) {
              (obj.geometry as THREE.BufferGeometry).dispose();
            }
            if ('material' in obj && obj.material) {
              const mat = obj.material as THREE.Material | THREE.Material[];
              const mats = Array.isArray(mat) ? mat : [mat];
              // 潜在问题修复：释放材质上的纹理（CanvasTexture等），避免内存泄漏
              mats.forEach((m) => {
                const anyMat = m as any;
                if (anyMat.map) anyMat.map.dispose();
                m.dispose();
              });
            }
          });

          renderer.dispose();
          if (container.contains(renderer.domElement)) {
            container.removeChild(renderer.domElement);
          }

          layerMeshesRef.current.clear();
          layerGroupsRef.current.clear();
          layerLabelsRef.current.clear();
          connectionLinesRef.current = [];
          layerPositionsRef.current.clear();
          particlesRef.current = null;
          controls.dispose();

          // 重置所有 ref，确保下次初始化能正确重建
          sceneRef.current = null;
          rendererRef.current = null;
          cameraRef.current = null;
          controlsRef.current = null;
        };
      } catch (err) {
        console.error('CNN3DViewer 初始化失败:', err);
        setInitStatus('error');
        return null;
      }
    };

    // 立即尝试初始化，如果容器还不可用则等待DOM就绪
    if (containerRef.current) {
      cleanupFn = initializeScene();
    } else {
      const observer = new MutationObserver((_, obs) => {
        if (containerRef.current) {
          obs.disconnect();
          cleanupFn = initializeScene();
        }
      });
      observer.observe(document.body, { childList: true, subtree: true });
      
      return () => {
        observer.disconnect();
        if (cleanupFn) cleanupFn();
      };
    }

    return () => {
      if (cleanupFn) cleanupFn();
    };
  }, [architecture, layers]);

  /* ---------- 视图模式切换 ---------- */
  useEffect(() => {
    if (!sceneRef.current) return;

    // 计算最大参数数用于归一化
    const maxParams = Math.max(...layers.map((l) => l.params || 1), 1);
    const logMaxParams = Math.log10(maxParams + 1);

    const layerIds = Array.from(layerMeshesRef.current.keys());
    layerIds.forEach((layerId, layerIdx) => {
      // 优先使用真实激活值，其次使用模拟激活值
      const realActArr = realActivations[layerId];
      const simActArr = activations?.[layerId];
      const meshes = layerMeshesRef.current.get(layerId);
      const layer = layers.find((l) => l.id === layerId);
      if (!meshes || !layer) return;

      const baseColor = LAYER_COLORS[layer.type];
      const paramNormalized = Math.min(1, Math.log10(layer.params + 1) / Math.max(logMaxParams, 1));

      meshes.forEach((mesh, idx) => {
        if (mesh.material instanceof THREE.MeshStandardMaterial) {
          if (viewMode === 'activation') {
            // 激活视图：用纯粹的冷暖色，对比度最高
            let activation: number;
            if (realActArr && realActArr.length > 0) {
              // 使用真实激活值
              activation = realActArr[idx % realActArr.length] ?? 0;
            } else if (simActArr && simActArr.length > 0) {
              // 使用模拟激活值
              activation = simActArr[idx % simActArr.length] ?? 0;
            } else {
              // 无激活值时使用模拟动态效果
              activation = Math.sin(layerIdx * 1.2 + idx * 0.7) * 0.7
                       + Math.sin(Date.now() * 0.003 + idx) * 0.3;
            }
            const color = getActivationColor(activation);
            mesh.material.color.copy(color);
            mesh.material.emissive.copy(color);
            mesh.material.emissiveIntensity = getActivationIntensity(activation);
            mesh.material.opacity = 0.9;
            mesh.material.metalness = 0.3;
            mesh.material.roughness = 0.4;
          } else if (viewMode === 'parameter') {
            // 参数视图：热图颜色（冷蓝→暖黄→炽热红）
            const color = getParameterColor(paramNormalized);
            mesh.material.color.copy(color);
            mesh.material.emissive.copy(color);
            mesh.material.emissiveIntensity = getParameterIntensity(paramNormalized);
            // 参数多的层更不透明
            mesh.material.opacity = 0.45 + paramNormalized * 0.5;
            mesh.material.metalness = 0.1 + paramNormalized * 0.5;
            mesh.material.roughness = 0.7 - paramNormalized * 0.4;
          } else {
            // 结构视图：使用层基础色，中等亮度，统一视觉风格
            mesh.material.color.copy(baseColor);
            mesh.material.emissive.copy(baseColor);
            mesh.material.emissiveIntensity = getStructureIntensity();
            mesh.material.opacity = 0.85;
            mesh.material.metalness = 0.25;
            mesh.material.roughness = 0.55;
          }
        }
      });
    });
  }, [viewMode, activations, realActivations, layers]);

  /* ---------- 3D导出处理 ---------- */
  const handleExport3D = useCallback(async (format: Export3DFormat | 'screenshot') => {
    if (isExporting) return;
    setIsExporting(true);
    setExportMenuOpen(false);

    try {
      const scene = sceneRef.current;
      const renderer = rendererRef.current;
      const camera = cameraRef.current;

      if (!scene || !renderer || !camera) {
        console.error('3D scene not ready for export');
        return;
      }

      const safeName = architecture.name.replace(/[^a-zA-Z0-9_-]/g, '_') || 'cnn_model';

      if (format === 'screenshot') {
        downloadScreenshot(renderer, scene, camera, safeName, '2k');
      } else {
        await export3DModel(scene, format, {
          filename: safeName,
          includeConnections: true,
          includeLabels: false,
          includeGrid: false,
          includeParticles: false,
        });
      }
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setIsExporting(false);
    }
  }, [isExporting, architecture.name]);

  /* ---------- 渲染 ---------- */
  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0d1018]">
      {/* 3D Canvas 容器 - Bug6修复：移除 min-w/min-h 限制，允许 flex 自适应宽度 */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* 加载状态 */}
      {initStatus === 'pending' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0d1018]">
          <div className="h-12 w-12 animate-spin rounded-full border-2 border-transparent border-t-primary" />
          <p className="mt-4 text-sm font-semibold text-muted-foreground">初始化 3D 场景...</p>
        </div>
      )}
      {initStatus === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0d1018]">
          <p className="text-sm text-red-400">3D 渲染失败</p>
        </div>
      )}

      {/* 居中重置画布按钮 - 左上角 */}
      {initStatus === 'success' && (
        <div className="absolute left-4 bottom-4 z-10 flex gap-2">
          <button
            onClick={() => {
              const camera = cameraRef.current;
              const controls = controlsRef.current;
              if (!camera || !controls) return;
              camera.position.set(14, 9, 17);
              controls.target.set(0, 0.8, 0);
              controls.update();
            }}
            title="居中重置画布"
            className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground backdrop-blur-sm transition-all hover:border-primary/30 hover:bg-primary/[0.06] hover:text-primary"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9V5a2 2 0 0 1 2-2h4M21 9V5a2 2 0 0 0-2-2h-4M3 15v4a2 2 0 0 0 2 2h4M21 15v4a2 2 0 0 1-2 2h-4" />
              <circle cx="12" cy="12" r="2" />
            </svg>
            居中重置
          </button>

          {/* 3D导出按钮 */}
          <div className="relative">
            <button
              onClick={() => setExportMenuOpen(!exportMenuOpen)}
              disabled={isExporting}
              title="导出3D模型"
              className="flex items-center gap-1.5 rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1.5 text-[10px] font-medium text-emerald-400 backdrop-blur-sm transition-all hover:border-emerald-500/40 hover:bg-emerald-500/20 disabled:opacity-50"
            >
              {isExporting ? (
                <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" strokeDasharray="32" strokeDashoffset="32" />
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
              )}
              导出3D
            </button>

            {exportMenuOpen && (
              <div className="absolute bottom-full left-0 mb-2 w-56 rounded-lg border border-white/[0.08] bg-[#0c0e17]/95 p-1 shadow-xl backdrop-blur-md">
                <div className="px-2 py-1.5 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/60">
                  3D模型格式 (科研用)
                </div>
                <button
                  onClick={() => handleExport3D('glb')}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] text-foreground/80 transition-colors hover:bg-primary/10 hover:text-primary"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded bg-emerald-500/20 text-[8px] font-bold text-emerald-400">GLB</span>
                  <div>
                    <div className="font-medium">GLB 模型</div>
                    <div className="text-[9px] text-muted-foreground">二进制glTF，Blender/3D Viewer</div>
                  </div>
                </button>
                <button
                  onClick={() => handleExport3D('gltf')}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] text-foreground/80 transition-colors hover:bg-primary/10 hover:text-primary"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded bg-blue-500/20 text-[8px] font-bold text-blue-400">GLTF</span>
                  <div>
                    <div className="font-medium">glTF JSON</div>
                    <div className="text-[9px] text-muted-foreground">Three.js/WebGL标准</div>
                  </div>
                </button>
                <button
                  onClick={() => handleExport3D('obj')}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] text-foreground/80 transition-colors hover:bg-primary/10 hover:text-primary"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded bg-orange-500/20 text-[8px] font-bold text-orange-400">OBJ</span>
                  <div>
                    <div className="font-medium">OBJ 模型</div>
                    <div className="text-[9px] text-muted-foreground">通用格式，Maya/3ds Max</div>
                  </div>
                </button>
                <button
                  onClick={() => handleExport3D('ply')}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] text-foreground/80 transition-colors hover:bg-primary/10 hover:text-primary"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded bg-purple-500/20 text-[8px] font-bold text-purple-400">PLY</span>
                  <div>
                    <div className="font-medium">PLY 点云</div>
                    <div className="text-[9px] text-muted-foreground">科研可视化/MeshLab</div>
                  </div>
                </button>
                <div className="my-1 h-px bg-white/[0.06]" />
                <div className="px-2 py-1.5 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/60">
                  高清截图 (论文用)
                </div>
                <button
                  onClick={() => handleExport3D('screenshot')}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] text-foreground/80 transition-colors hover:bg-primary/10 hover:text-primary"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded bg-pink-500/20 text-[8px] font-bold text-pink-400">PNG</span>
                  <div>
                    <div className="font-medium">2K 高清截图</div>
                    <div className="text-[9px] text-muted-foreground">2560×1440 PNG</div>
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 点击外部关闭导出菜单 */}
      {exportMenuOpen && (
        <div
          className="absolute inset-0 z-[5]"
          onClick={() => setExportMenuOpen(false)}
        />
      )}

      {/* 状态信息 - 左上角 */}
      {initStatus === 'success' && (
        <div className="pointer-events-none absolute left-4 top-4 flex flex-col gap-2">
          <div className="rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 px-3 py-1.5 backdrop-blur-sm">
            <div className="flex items-center gap-2">
              <span className={`size-1.5 rounded-full ${isPlaying ? 'animate-pulse bg-emerald-400' : 'bg-muted-foreground'}`} />
              <span className="font-mono text-[11px] font-semibold text-foreground/80">
                {architecture.name}
              </span>
            </div>
          </div>

          <div className="rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 px-3 py-1.5 backdrop-blur-sm">
            <span className="font-mono text-[11px] text-muted-foreground">
              Step <span className="font-bold text-foreground">{step}</span>
            </span>
          </div>

          {/* 激活层指示 */}
          {activeLayerId && isPlaying && (
            <div className="rounded-lg border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 backdrop-blur-sm">
              <span className="font-mono text-[11px] font-semibold text-emerald-400">
                ▶ {layers.find((l) => l.id === activeLayerId)?.name}
              </span>
            </div>
          )}

          {/* 选中层指示 */}
          {effectiveSelectedId && (
            <div className="rounded-lg border border-primary/30 bg-primary/10 px-3 py-1.5 backdrop-blur-sm">
              <span className="font-mono text-[11px] font-semibold text-primary">
                ◉ {layers.find((l) => l.id === effectiveSelectedId)?.name}
              </span>
            </div>
          )}

          {/* Hover 提示 */}
          {hoveredLayerName && (
            <div className="rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 backdrop-blur-sm">
              <span className="font-mono text-[11px] text-foreground/70">
                {hoveredLayerName}
              </span>
            </div>
          )}
        </div>
      )}

      {/* 视图模式指示 - 右上角 */}
      {initStatus === 'success' && (
        <div className="pointer-events-none absolute right-4 top-4 rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 px-3 py-1.5 backdrop-blur-sm">
          <span className="font-mono text-[10px] text-muted-foreground">
            Mode
          </span>
          <span className="ml-2 font-mono text-[11px] font-bold text-foreground/80 capitalize">
            {viewMode}
          </span>
        </div>
      )}

      {/* 动态图例 - 右下角（可折叠）- Bug5修复：上移至 bottom-24 避免与播放控制条重叠，z-10 低于控制条 z-20 */}
      {initStatus === 'success' && viewMode === 'structure' && (
        <div className="absolute bottom-16 right-4 z-10 rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 p-2 backdrop-blur-sm max-w-[280px]">
          <button
            onClick={() => setLegendCollapsed((prev) => !prev)}
            className="flex w-full items-center justify-center gap-1.5 pb-1 text-[9px] uppercase tracking-wider text-muted-foreground/60 transition-colors hover:text-foreground/80"
          >
            <span>Layer Types</span>
            <svg
              width="10"
              height="10"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{ transform: legendCollapsed ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 200ms ease' }}
            >
              <path d="M18 15l-6-6-6 6" />
            </svg>
          </button>
          {!legendCollapsed && (
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(LAYER_COLORS_HEX) as LayerCategory[]).map((cat) => (
                <div key={cat} className="flex items-center gap-1.5 rounded bg-white/[0.02] px-1.5 py-0.5">
                  <span
                    className="size-1.5 rounded-full"
                    style={{ backgroundColor: LAYER_COLORS_HEX[cat] }}
                  />
                  <span className="font-mono text-[10px] text-foreground/70 capitalize">
                    {cat}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {initStatus === 'success' && viewMode === 'activation' && (
        <div className="pointer-events-none absolute bottom-16 right-4 z-10 rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 p-2.5 backdrop-blur-sm">
          <span className="mb-1.5 block text-center text-[9px] uppercase tracking-wider text-muted-foreground/60">
            Activation Scale
          </span>
          <div className="flex h-2 w-40 rounded-full bg-gradient-to-r from-[#4a0010] via-[#555555] to-[#cfffff]" />
          <div className="mt-1 flex justify-between text-[9px] text-muted-foreground">
            <span>Negative</span>
            <span>Zero</span>
            <span>Positive</span>
          </div>
        </div>
      )}

      {initStatus === 'success' && viewMode === 'parameter' && (
        <div className="pointer-events-none absolute bottom-16 right-4 z-10 rounded-lg border border-white/[0.08] bg-[#0c0e17]/85 p-2.5 backdrop-blur-sm">
          <span className="mb-1.5 block text-center text-[9px] uppercase tracking-wider text-muted-foreground/60">
            Parameter Heatmap
          </span>
          <div className="flex h-2 w-40 rounded-full bg-gradient-to-r from-[#1a237e] via-[#00e5ff] via-[#ffea00] to-[#ff3d00]" />
          <div className="mt-1 flex justify-between text-[9px] text-muted-foreground">
            <span>Low</span>
            <span>Mid</span>
            <span>High</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- 默认导出（兼容现有导入） ---------- */
export default CNN3DViewer;
