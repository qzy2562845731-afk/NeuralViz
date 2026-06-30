/**
 * 3D 模型导出工具
 * 支持 GLB (二进制glTF) / GLTF (JSON glTF) / OBJ 格式导出
 * 用于科研论文3D可视化、Blender/ParaView等工具二次编辑
 */
import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js';
import { PLYExporter } from 'three/examples/jsm/exporters/PLYExporter.js';

export type Export3DFormat = 'glb' | 'gltf' | 'obj' | 'ply';

export interface Export3DOptions {
  /** 是否包含粒子系统(默认false) */
  includeParticles?: boolean;
  /** 是否包含连接线(默认true) */
  includeConnections?: boolean;
  /** 是否包含标签Sprite(默认false，标签是Canvas纹理在外部软件中无法显示) */
  includeLabels?: boolean;
  /** 是否包含网格地面(默认false) */
  includeGrid?: boolean;
  /** 文件名(不含扩展名) */
  filename?: string;
}

const DEFAULT_OPTIONS: Required<Export3DOptions> = {
  includeParticles: false,
  includeConnections: true,
  includeLabels: false,
  includeGrid: false,
  filename: 'cnn_architecture',
};

/**
 * 准备导出场景：克隆并过滤不需要的对象
 */
function prepareSceneForExport(
  scene: THREE.Scene,
  options: Required<Export3DOptions>
): THREE.Group {
  const exportGroup = new THREE.Group();
  exportGroup.name = 'CNN_Architecture';

  scene.traverse((obj) => {
    if (obj === scene) return;

    if (obj instanceof THREE.Points && !options.includeParticles) return;
    if (obj instanceof THREE.Line && !options.includeConnections) {
      if (obj.parent?.type !== 'Group') return;
    }
    if (obj instanceof THREE.Sprite && !options.includeLabels) return;
    if (obj instanceof THREE.GridHelper && !options.includeGrid) return;

    if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Points) {
      const clone = obj.clone();
      if (clone.material) {
        if (Array.isArray(clone.material)) {
          clone.material = clone.material.map((m) => {
            const mat = m.clone();
            mat.transparent = false;
            mat.opacity = 1;
            return mat;
          });
        } else {
          const mat = clone.material.clone();
          mat.transparent = false;
          mat.opacity = 1;
          clone.material = mat;
        }
      }
      exportGroup.add(clone);
    }
  });

  return exportGroup;
}

/**
 * 触发浏览器下载
 */
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

/**
 * 导出为 GLB 格式 (推荐，二进制，包含材质，体积小)
 * GLB是glTF的二进制版本，可直接在Blender 3.0+、Three.js Editor、
 * Microsoft 3D Viewer、ParaView等工具中打开
 */
export function exportGLB(
  scene: THREE.Scene,
  options: Export3DOptions = {}
): Promise<void> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  return new Promise((resolve, reject) => {
    try {
      const gltfExporter = new GLTFExporter();
      const exportObj = prepareSceneForExport(scene, opts);

      gltfExporter.parse(
        exportObj,
        (result) => {
          if (result instanceof ArrayBuffer) {
            const blob = new Blob([result], { type: 'model/gltf-binary' });
            triggerDownload(blob, `${opts.filename}.glb`);
          } else {
            reject(new Error('GLB 导出结果格式错误'));
            return;
          }
          resolve();
        },
        (error) => {
          reject(error);
        },
        { binary: true }
      );
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * 导出为 GLTF 格式 (JSON格式，包含外部bin)
 */
export function exportGLTF(
  scene: THREE.Scene,
  options: Export3DOptions = {}
): Promise<void> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  return new Promise((resolve, reject) => {
    try {
      const gltfExporter = new GLTFExporter();
      const exportObj = prepareSceneForExport(scene, opts);

      gltfExporter.parse(
        exportObj,
        (result) => {
          if (typeof result === 'object' && !(result instanceof ArrayBuffer)) {
            const jsonStr = JSON.stringify(result, null, 2);
            const blob = new Blob([jsonStr], { type: 'model/gltf+json' });
            triggerDownload(blob, `${opts.filename}.gltf`);
          } else {
            reject(new Error('GLTF 导出结果格式错误'));
            return;
          }
          resolve();
        },
        (error) => {
          reject(error);
        },
        { binary: false }
      );
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * 导出为 OBJ 格式 (通用3D格式，几乎所有3D软件都支持)
 * 注意：OBJ不支持材质颜色和层级结构，仅导出几何体
 */
export function exportOBJ(
  scene: THREE.Scene,
  options: Export3DOptions = {}
): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const objExporter = new OBJExporter();
  const exportObj = prepareSceneForExport(scene, opts);
  const result = objExporter.parse(exportObj);
  const blob = new Blob([result], { type: 'text/plain' });
  triggerDownload(blob, `${opts.filename}.obj`);
}

/**
 * 导出为 PLY 格式 (点云/多边形格式，常用于科研可视化)
 */
export function exportPLY(
  scene: THREE.Scene,
  options: Export3DOptions = {}
): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const plyExporter = new PLYExporter();
  const exportObj = prepareSceneForExport(scene, opts);
  plyExporter.parse(
    exportObj,
    (result: string | ArrayBuffer) => {
      const blob = new Blob([result], { type: 'application/octet-stream' });
      triggerDownload(blob, `${opts.filename}.ply`);
    }
  );
}

/**
 * 导出3D模型（统一接口）
 */
export function export3DModel(
  scene: THREE.Scene,
  format: Export3DFormat,
  options: Export3DOptions = {}
): Promise<void> | void {
  switch (format) {
    case 'glb':
      return exportGLB(scene, options);
    case 'gltf':
      return exportGLTF(scene, options);
    case 'obj':
      return exportOBJ(scene, options);
    case 'ply':
      return exportPLY(scene, options);
    default:
      throw new Error(`不支持的导出格式: ${format}`);
  }
}

/**
 * 获取当前Three.js场景的截图(高分辨率PNG)
 * 用于论文插图
 */
export function captureScreenshot(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  width: number = 3840,
  height: number = 2160
): string {
  const originalSize = renderer.getSize(new THREE.Vector2());
  renderer.setSize(width, height);
  renderer.render(scene, camera);
  const dataURL = renderer.domElement.toDataURL('image/png');
  renderer.setSize(originalSize.x, originalSize.y);
  renderer.render(scene, camera);
  return dataURL;
}

/**
 * 下载截图为PNG文件
 */
export function downloadScreenshot(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  filename: string = 'cnn_3d_view',
  resolution: '4k' | '2k' | '1080p' = '2k'
): void {
  const sizes = {
    '4k': [3840, 2160],
    '2k': [2560, 1440],
    '1080p': [1920, 1080],
  } as const;
  const [w, h] = sizes[resolution];
  const dataURL = captureScreenshot(renderer, scene, camera, w, h);
  const link = document.createElement('a');
  link.download = `${filename}_${resolution}_${Date.now()}.png`;
  link.href = dataURL;
  link.click();
}
