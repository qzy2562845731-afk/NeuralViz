import { useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';

/* ============================================
   NeuralNetworkBackground — 动态神经网络背景
   - 漂浮的神经元节点
   - 节点间的动态连接线
   - 数据脉冲沿连线流动
   - 鼠标交互：粒子轻微避让
   - 多层视差，营造深度感
   ============================================ */

export function NeuralNetworkBackground() {
  const containerRef = useRef<HTMLDivElement>(null);

  // 配置参数
  const config = useMemo(
    () => ({
      nodeCount: 80,
      connectionDistance: 180,
      maxConnections: 4,
      baseSpeed: 0.0004,
      pulseSpeed: 1.2,
      colors: {
        node: new THREE.Color('#00d4ff'),
        nodeGlow: new THREE.Color('#00a8e8'),
        line: new THREE.Color('#0a7a8a'),
        pulse: new THREE.Color('#ffffff'),
        bg: new THREE.Color('#030712'),
      },
    }),
    []
  );

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = window.innerWidth;
    const height = window.innerHeight;

    // 创建场景
    const scene = new THREE.Scene();
    scene.background = config.colors.bg;

    // 相机
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 2000);
    camera.position.z = 500;

    // 渲染器
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x030712, 1);
    container.appendChild(renderer.domElement);

    // 节点数据
    const nodes: Array<{
      x: number;
      y: number;
      z: number;
      vx: number;
      vy: number;
      vz: number;
      radius: number;
      phase: number;
    }> = [];

    for (let i = 0; i < config.nodeCount; i++) {
      nodes.push({
        x: (Math.random() - 0.5) * width * 1.5,
        y: (Math.random() - 0.5) * height * 1.5,
        z: (Math.random() - 0.5) * 600,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        vz: (Math.random() - 0.5) * 0.15,
        radius: 2 + Math.random() * 3,
        phase: Math.random() * Math.PI * 2,
      });
    }

    // 创建节点材质和几何体
    const nodeGeometry = new THREE.BufferGeometry();
    const nodePositions = new Float32Array(config.nodeCount * 3);
    const nodeSizes = new Float32Array(config.nodeCount);
    const nodeColors = new Float32Array(config.nodeCount * 3);

    for (let i = 0; i < config.nodeCount; i++) {
      nodePositions[i * 3] = nodes[i].x;
      nodePositions[i * 3 + 1] = nodes[i].y;
      nodePositions[i * 3 + 2] = nodes[i].z;
      nodeSizes[i] = nodes[i].radius;
      nodeColors[i * 3] = config.colors.node.r;
      nodeColors[i * 3 + 1] = config.colors.node.g;
      nodeColors[i * 3 + 2] = config.colors.node.b;
    }

    nodeGeometry.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
    nodeGeometry.setAttribute('size', new THREE.BufferAttribute(nodeSizes, 1));
    nodeGeometry.setAttribute('color', new THREE.BufferAttribute(nodeColors, 3));

    // 自定义节点着色器
    const nodeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        uniform float uTime;
        uniform float uPixelRatio;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          float pulse = 1.0 + 0.3 * sin(uTime * 2.0 + position.x * 0.01);
          gl_PointSize = size * uPixelRatio * pulse * (400.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          float dist = length(gl_PointCoord - vec2(0.5));
          if (dist > 0.5) discard;
          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
          float glow = exp(-dist * 4.0) * 0.6;
          gl_FragColor = vec4(vColor + glow, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const nodePoints = new THREE.Points(nodeGeometry, nodeMaterial);
    scene.add(nodePoints);

    // 连线和脉冲
    const maxLines = config.nodeCount * config.maxConnections;
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(maxLines * 2 * 3);
    const lineColors = new Float32Array(maxLines * 2 * 3);
    const lineOpacities = new Float32Array(maxLines * 2);

    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    lineGeometry.setAttribute('opacity', new THREE.BufferAttribute(lineOpacities, 1));

    const lineMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
      },
      vertexShader: `
        attribute float opacity;
        varying float vOpacity;
        varying vec3 vColor;
        void main() {
          vOpacity = opacity;
          vColor = color;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying float vOpacity;
        varying vec3 vColor;
        void main() {
          gl_FragColor = vec4(vColor, vOpacity);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lineSegments);

    // 脉冲粒子
    const pulseCount = 30;
    const pulseGeometry = new THREE.BufferGeometry();
    const pulsePositions = new Float32Array(pulseCount * 3);
    const pulseProgress = new Float32Array(pulseCount);
    const pulseTargets = new Int32Array(pulseCount * 2);

    for (let i = 0; i < pulseCount; i++) {
      pulseProgress[i] = -1;
    }

    pulseGeometry.setAttribute('position', new THREE.BufferAttribute(pulsePositions, 3));

    const pulseMaterial = new THREE.PointsMaterial({
      color: config.colors.pulse,
      size: 6,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const pulsePoints = new THREE.Points(pulseGeometry, pulseMaterial);
    scene.add(pulsePoints);

    // 鼠标交互
    const mouse = new THREE.Vector2(0, 0);
    const targetMouse = new THREE.Vector2(0, 0);

    const handleMouseMove = (e: MouseEvent) => {
      targetMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      targetMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });

    // 动画循环
    let frameId: number;
    let time = 0;

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      time += config.baseSpeed;

      // 平滑鼠标
      mouse.x += (targetMouse.x - mouse.x) * 0.05;
      mouse.y += (targetMouse.y - mouse.y) * 0.05;

      // 更新节点位置
      const positions = nodeGeometry.attributes.position.array as Float32Array;

      for (let i = 0; i < config.nodeCount; i++) {
        const node = nodes[i];

        // 基础移动
        node.x += node.vx;
        node.y += node.vy;
        node.z += node.vz;

        // 边界回弹
        const boundX = width * 0.8;
        const boundY = height * 0.8;
        const boundZ = 400;

        if (Math.abs(node.x) > boundX) node.vx *= -1;
        if (Math.abs(node.y) > boundY) node.vy *= -1;
        if (Math.abs(node.z) > boundZ) node.vz *= -1;

        // 鼠标避让
        const mouseInfluenceX = mouse.x * width * 0.3;
        const mouseInfluenceY = mouse.y * height * 0.3;
        const dx = node.x - mouseInfluenceX;
        const dy = node.y - mouseInfluenceY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 200) {
          const force = ((200 - dist) / 200) * 0.5;
          node.x += (dx / dist) * force;
          node.y += (dy / dist) * force;
        }

        // 轻微漂浮
        node.x += Math.sin(time + node.phase) * 0.1;
        node.y += Math.cos(time + node.phase) * 0.1;

        positions[i * 3] = node.x;
        positions[i * 3 + 1] = node.y;
        positions[i * 3 + 2] = node.z;
      }

      nodeGeometry.attributes.position.needsUpdate = true;
      nodeMaterial.uniforms.uTime.value = time * 10;

      // 更新连线
      let lineIndex = 0;
      const linePos = lineGeometry.attributes.position.array as Float32Array;
      const lineCol = lineGeometry.attributes.color.array as Float32Array;
      const lineOp = lineGeometry.attributes.opacity.array as Float32Array;

      for (let i = 0; i < config.nodeCount; i++) {
        let connections = 0;
        for (let j = i + 1; j < config.nodeCount && connections < config.maxConnections; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dz = nodes[i].z - nodes[j].z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < config.connectionDistance) {
            const opacity = (1 - dist / config.connectionDistance) * 0.35;

            linePos[lineIndex * 6] = nodes[i].x;
            linePos[lineIndex * 6 + 1] = nodes[i].y;
            linePos[lineIndex * 6 + 2] = nodes[i].z;
            linePos[lineIndex * 6 + 3] = nodes[j].x;
            linePos[lineIndex * 6 + 4] = nodes[j].y;
            linePos[lineIndex * 6 + 5] = nodes[j].z;

            const pulse = 0.5 + 0.5 * Math.sin(time * 3 + i * 0.5 + j * 0.3);
            const r = config.colors.line.r + config.colors.node.r * pulse * 0.3;
            const g = config.colors.line.g + config.colors.node.g * pulse * 0.3;
            const b = config.colors.line.b + config.colors.node.b * pulse * 0.3;

            lineCol[lineIndex * 6] = r;
            lineCol[lineIndex * 6 + 1] = g;
            lineCol[lineIndex * 6 + 2] = b;
            lineCol[lineIndex * 6 + 3] = r;
            lineCol[lineIndex * 6 + 4] = g;
            lineCol[lineIndex * 6 + 5] = b;

            lineOp[lineIndex * 2] = opacity;
            lineOp[lineIndex * 2 + 1] = opacity;

            lineIndex++;
            connections++;
          }
        }
      }

      // 清空未使用的连线
      for (let i = lineIndex; i < maxLines; i++) {
        lineOp[i * 2] = 0;
        lineOp[i * 2 + 1] = 0;
      }

      lineGeometry.attributes.position.needsUpdate = true;
      lineGeometry.attributes.color.needsUpdate = true;
      lineGeometry.attributes.opacity.needsUpdate = true;
      lineMaterial.uniforms.uTime.value = time * 10;

      // 更新脉冲
      const pulsePos = pulseGeometry.attributes.position.array as Float32Array;
      for (let i = 0; i < pulseCount; i++) {
        if (pulseProgress[i] < 0) {
          // 生成新脉冲
          if (Math.random() < 0.02) {
            const sourceIdx = Math.floor(Math.random() * config.nodeCount);
            let targetIdx = Math.floor(Math.random() * config.nodeCount);
            while (targetIdx === sourceIdx) {
              targetIdx = Math.floor(Math.random() * config.nodeCount);
            }
            pulseProgress[i] = 0;
            pulseTargets[i * 2] = sourceIdx;
            pulseTargets[i * 2 + 1] = targetIdx;
          }
        } else {
          pulseProgress[i] += 0.008 * config.pulseSpeed;
          if (pulseProgress[i] >= 1) {
            pulseProgress[i] = -1;
            pulsePos[i * 3] = 0;
            pulsePos[i * 3 + 1] = 0;
            pulsePos[i * 3 + 2] = -1000;
          } else {
            const sourceIdx = pulseTargets[i * 2];
            const targetIdx = pulseTargets[i * 2 + 1];
            const source = nodes[sourceIdx];
            const target = nodes[targetIdx];

            pulsePos[i * 3] = source.x + (target.x - source.x) * pulseProgress[i];
            pulsePos[i * 3 + 1] = source.y + (target.y - source.y) * pulseProgress[i];
            pulsePos[i * 3 + 2] = source.z + (target.z - source.z) * pulseProgress[i];
          }
        }
      }
      pulseGeometry.attributes.position.needsUpdate = true;

      // 缓慢旋转相机
      camera.position.x = Math.sin(time * 0.2) * 30;
      camera.position.y = Math.cos(time * 0.15) * 20;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    animate();

    // 窗口大小调整
    const handleResize = () => {
      const newWidth = window.innerWidth;
      const newHeight = window.innerHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
      nodeMaterial.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [config]);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 -z-10"
      style={{
        background: 'radial-gradient(ellipse at 50% 0%, #0a1628 0%, #030712 50%, #02040a 100%)',
      }}
    />
  );
}

export default NeuralNetworkBackground;
