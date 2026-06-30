import * as THREE from 'three';

/* ============================================
   OrbitControls 轨道相机控制 — 精简正确版
   支持：
   - 鼠标左键拖拽：旋转相机绕 target
   - 滚轮：缩放距离
   - 右键拖拽：平移 target
   无外部依赖，无框架要求
   ============================================ */

export class OrbitControls {
  // 目标点
  public target: THREE.Vector3 = new THREE.Vector3();

  // 相机与 DOM
  public camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  public domElement: HTMLElement;

  // 距离限制
  public minDistance: number = 5;
  public maxDistance: number = 50;

  // 角度限制 (弧度)
  public minPolarAngle: number = 0.1;
  public maxPolarAngle: number = Math.PI / 2 - 0.05; // 限制看不到底半球

  // 阻尼
  public enableDamping: boolean = true;
  public dampingFactor: number = 0.08;

  // 自动旋转
  public autoRotate: boolean = false;
  public autoRotateSpeed: number = 0.5;

  // 功能开关
  public enabled: boolean = true;
  public enableZoom: boolean = true;
  public enableRotate: boolean = true;
  public enablePan: boolean = true;

  // 内部球坐标 (用于动画)
  private spherical: THREE.Spherical = new THREE.Spherical();
  private sphericalDelta: THREE.Spherical = new THREE.Spherical();
  private scale: number = 1;
  private panOffset: THREE.Vector3 = new THREE.Vector3();

  // 状态
  private STATE = { NONE: -1, ROTATE: 0, PAN: 1 } as const;
  private state: number = this.STATE.NONE;
  private rotateStart: THREE.Vector2 = new THREE.Vector2();
  private rotateEnd: THREE.Vector2 = new THREE.Vector2();
  private rotateDelta: THREE.Vector2 = new THREE.Vector2();
  private panStart: THREE.Vector2 = new THREE.Vector2();
  private panEnd: THREE.Vector2 = new THREE.Vector2();
  private panDelta: THREE.Vector2 = new THREE.Vector2();

  // 临时
  private offset: THREE.Vector3 = new THREE.Vector3();
  private quat: THREE.Quaternion = new THREE.Quaternion();
  private quatInverse: THREE.Quaternion = new THREE.Quaternion();
  private lastPosition: THREE.Vector3 = new THREE.Vector3();
  private lastQuaternion: THREE.Quaternion = new THREE.Quaternion();

  // 事件回调
  public onUpdate?: () => void;

  constructor(
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
    domElement: HTMLElement
  ) {
    this.camera = camera;
    this.domElement = domElement;

    // 初始化相机姿态
    this.updateSphericalFromCamera();

    // 注册事件
    this.domElement.addEventListener('contextmenu', this.onContextMenu);
    this.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.domElement.addEventListener('wheel', this.onMouseWheel, { passive: false });

    this.update();
  }

  public dispose(): void {
    this.domElement.removeEventListener('contextmenu', this.onContextMenu);
    this.domElement.removeEventListener('pointerdown', this.onPointerDown);
    this.domElement.removeEventListener('wheel', this.onMouseWheel);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
  }

  public reset(): void {
    this.target.set(0, 0, 0);
    this.updateSphericalFromCamera();
    this.panOffset.set(0, 0, 0);
    this.scale = 1;
    this.update();
  }

  /* ---------- 公开设置 target 并同步球坐标 ---------- */
  public setTarget(x: number, y: number, z: number): void {
    this.target.set(x, y, z);
    this.updateSphericalFromCamera();
  }

  /* ---------- 根据当前相机位置 → 球坐标 ---------- */
  private updateSphericalFromCamera(): void {
    this.offset.copy(this.camera.position).sub(this.target);
    this.quat.setFromUnitVectors(this.camera.up, new THREE.Vector3(0, 1, 0));
    this.quatInverse.copy(this.quat).invert();
    this.offset.applyQuaternion(this.quat);
    this.spherical.setFromVector3(this.offset);
  }

  /* ---------- 主循环：每帧调用 ---------- */
  public update(): void {
    // 自动旋转
    if (this.autoRotate && this.state === this.STATE.NONE) {
      this.sphericalDelta.theta -= this.getAutoRotationAngle();
    }

    // ---- 旋转增量应用到 spherical ----
    this.spherical.theta += this.sphericalDelta.theta;
    this.spherical.phi += this.sphericalDelta.phi;

    // ---- 限制极角 (0 < phi < PI) ----
    this.spherical.phi = Math.max(
      this.minPolarAngle,
      Math.min(this.maxPolarAngle, this.spherical.phi)
    );

    // ---- 限制半径 ----
    this.spherical.radius *= this.scale;
    this.spherical.radius = Math.max(
      this.minDistance,
      Math.min(this.maxDistance, this.spherical.radius)
    );
    this.scale = 1;

    // ---- 从球坐标重建 offset ----
    this.offset.setFromSpherical(this.spherical);

    // ---- 旋转回 camera-up 坐标系 ----
    this.offset.applyQuaternion(this.quatInverse);

    // ---- 平移 target ----
    this.target.add(this.panOffset);
    this.panOffset.multiplyScalar(this.enableDamping ? 1 - this.dampingFactor : 0);

    // ---- 更新相机 ----
    this.camera.position.copy(this.target).add(this.offset);
    this.camera.lookAt(this.target);

    // ---- 阻尼衰减 ----
    if (this.enableDamping) {
      this.sphericalDelta.theta *= 1 - this.dampingFactor;
      this.sphericalDelta.phi *= 1 - this.dampingFactor;
      this.sphericalDelta.radius *= 1 - this.dampingFactor;
    } else {
      this.sphericalDelta.set(0, 0, 0);
    }

    // ---- 触发更新事件 ----
    if (
      this.lastPosition.distanceToSquared(this.camera.position) > 1e-8 ||
      8 * (1 - this.lastQuaternion.dot(this.camera.quaternion)) > 1e-8
    ) {
      this.lastPosition.copy(this.camera.position);
      this.lastQuaternion.copy(this.camera.quaternion);
      if (this.onUpdate) this.onUpdate();
    }
  }

  /* ---------- 聚焦到目标点 ---------- */
  public focusOn(targetPos: THREE.Vector3, duration: number = 600): void {
    const startTarget = this.target.clone();
    const startTime = performance.now();

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const t = Math.min(1, elapsed / duration);
      const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      this.target.lerpVectors(startTarget, targetPos, eased);
      this.update();

      if (t < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }

  /* ---------- 内部：自动旋转角度 ---------- */
  private getAutoRotationAngle(): number {
    return (2 * Math.PI) / 60 / 60 * this.autoRotateSpeed;
  }

  /* ---------- 事件处理 ---------- */
  private onContextMenu = (event: MouseEvent): void => {
    if (this.enabled) event.preventDefault();
  };

  private onPointerDown = (event: PointerEvent): void => {
    if (!this.enabled) return;

    switch (event.button) {
      case 0:
        if (!this.enableRotate) return;
        this.state = this.STATE.ROTATE;
        this.rotateStart.set(event.clientX, event.clientY);
        break;
      case 1:
      case 2:
        if (!this.enablePan) return;
        this.state = this.STATE.PAN;
        this.panStart.set(event.clientX, event.clientY);
        break;
    }

    if (this.state !== this.STATE.NONE) {
      window.addEventListener('pointermove', this.onPointerMove);
      window.addEventListener('pointerup', this.onPointerUp);
    }
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (!this.enabled) return;

    if (this.state === this.STATE.ROTATE && this.enableRotate) {
      this.rotateEnd.set(event.clientX, event.clientY);
      this.rotateDelta.subVectors(this.rotateEnd, this.rotateStart);
      // 转换像素 → 弧度（经验值）
      this.sphericalDelta.theta -= (2 * Math.PI * this.rotateDelta.x) / this.domElement.clientHeight;
      this.sphericalDelta.phi -= (2 * Math.PI * this.rotateDelta.y) / this.domElement.clientHeight;
      this.rotateStart.copy(this.rotateEnd);
      this.update();
    } else if (this.state === this.STATE.PAN && this.enablePan) {
      this.panEnd.set(event.clientX, event.clientY);
      this.panDelta.subVectors(this.panEnd, this.panStart).multiplyScalar(0.02);

      const distance = this.camera.position.distanceTo(this.target);
      const te = this.camera.matrix.elements;

      // X 方向平移
      const panX = new THREE.Vector3(te[0], te[1], te[2]);
      panX.multiplyScalar(-this.panDelta.x * distance / this.domElement.clientHeight * 2);

      // Y 方向平移
      const panY = new THREE.Vector3(te[4], te[5], te[6]);
      panY.multiplyScalar(this.panDelta.y * distance / this.domElement.clientHeight * 2);

      this.panOffset.add(panX).add(panY);
      this.panStart.copy(this.panEnd);
      this.update();
    }
  };

  private onPointerUp = (): void => {
    this.state = this.STATE.NONE;
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
  };

  private onMouseWheel = (event: WheelEvent): void => {
    if (!this.enabled || !this.enableZoom) return;
    event.preventDefault();

    if (event.deltaY < 0) {
      this.scale *= 0.95;  // 放大 → 减小半径
    } else if (event.deltaY > 0) {
      this.scale *= 1.05;  // 缩小 → 增大半径
    }

    this.update();
  };
}
