import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { Download, Layers } from 'lucide-react';

const MIME_BY_FORMAT = {
  glb: 'model/gltf-binary',
  gltf: 'model/gltf+json',
  obj: 'text/plain',
  fbx: 'application/octet-stream',
  ply: 'application/octet-stream',
  stl: 'model/stl',
};

const MESH_EXTENSIONS = ['glb', 'gltf', 'obj', 'fbx', 'ply', 'stl'];

export function guessMeshFormat(source) {
  if (!source) return null;
  const clean = source.split('?')[0].split('#')[0];
  const ext = clean.split('.').pop()?.toLowerCase();
  return MESH_EXTENSIONS.includes(ext) ? ext : null;
}

export function isMeshResource(type, name) {
  if (type && type.startsWith('model/')) return true;
  return Boolean(guessMeshFormat(name));
}

function base64ToArrayBuffer(base64) {
  const clean = base64.includes(',') ? base64.split(',')[1] : base64;
  const binary = atob(clean);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function defaultMaterial() {
  return new THREE.MeshStandardMaterial({ color: 0x9db4ff, metalness: 0.1, roughness: 0.7 });
}

function frameCamera(object, camera, controls) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const fov = camera.fov * (Math.PI / 180);
  let distance = Math.abs(maxDim / 2 / Math.tan(fov / 2));
  distance *= 1.6;
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.position.set(center.x + distance * 0.4, center.y + distance * 0.3, center.z + distance);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

export default function Mesh3DViewer({ base64, url, format }) {
  const containerRef = useRef(null);
  const [wireframe, setWireframe] = useState(false);
  const wireframeRef = useRef(wireframe);
  const loadedObjectRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => { wireframeRef.current = wireframe; applyWireframe(loadedObjectRef.current, wireframe); }, [wireframe]);

  function applyWireframe(object, value) {
    if (!object) return;
    object.traverse(child => {
      if (!child.isMesh) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach(mat => { if (mat) mat.wireframe = value; });
    });
  }

  useEffect(() => {
    const container = containerRef.current;
    if (!container || (!base64 && !url)) return undefined;
    setError(null);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090e);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / Math.max(1, container.clientHeight), 0.01, 1000);
    camera.position.set(2, 2, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1a1a2e, 1.1);
    scene.add(hemiLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 5, 4);
    scene.add(dirLight);
    scene.add(new THREE.AmbientLight(0x404040, 0.6));

    const grid = new THREE.GridHelper(10, 20, 0x2a3550, 0x1e293b);
    grid.position.y = -0.001;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    let animationFrameId = null;
    let disposed = false;
    const disposables = [];
    let loadedObject = null;

    const onObjectReady = (object) => {
      if (disposed) return;
      applyWireframe(object, wireframeRef.current);
      scene.add(object);
      loadedObject = object;
      loadedObjectRef.current = object;
      frameCamera(object, camera, controls);
    };

    const fmt = (format || guessMeshFormat(url) || 'glb').toLowerCase();

    const processBuffer = (buffer) => {
      if (disposed) return;
      if (fmt === 'glb' || fmt === 'gltf') {
        const loader = new GLTFLoader();
        loader.parse(buffer, '', (gltf) => onObjectReady(gltf.scene), (err) => setError(String(err?.message || err)));
      } else if (fmt === 'obj') {
        const text = new TextDecoder('utf-8').decode(buffer);
        const loader = new OBJLoader();
        const object = loader.parse(text);
        object.traverse(child => {
          if (child.isMesh && (!child.material || (Array.isArray(child.material) && child.material.length === 0))) {
            child.material = defaultMaterial();
          }
        });
        onObjectReady(object);
      } else if (fmt === 'fbx') {
        const loader = new FBXLoader();
        const object = loader.parse(buffer, '');
        object.traverse(child => {
          if (child.isMesh && !child.material) child.material = defaultMaterial();
        });
        onObjectReady(object);
      } else if (fmt === 'ply') {
        const loader = new PLYLoader();
        const geometry = loader.parse(buffer);
        geometry.computeVertexNormals();
        disposables.push(geometry);
        const mesh = new THREE.Mesh(geometry, defaultMaterial());
        onObjectReady(mesh);
      } else if (fmt === 'stl') {
        const loader = new STLLoader();
        const geometry = loader.parse(buffer);
        geometry.computeVertexNormals();
        disposables.push(geometry);
        const mesh = new THREE.Mesh(geometry, defaultMaterial());
        onObjectReady(mesh);
      } else {
        setError(`Unsupported mesh format: ${fmt}`);
      }
    };

    (async () => {
      try {
        let buffer;
        if (base64) {
          buffer = base64ToArrayBuffer(base64);
        } else {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          buffer = await res.arrayBuffer();
        }
        if (disposed) return;
        processBuffer(buffer);
      } catch (e) {
        if (!disposed) setError(String(e?.message || e));
      }
    })();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resizeObserver = new ResizeObserver(() => {
      if (!container.clientWidth || !container.clientHeight) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);

    return () => {
      disposed = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      controls.dispose();
      if (loadedObject) {
        loadedObject.traverse(child => {
          if (child.isMesh) {
            child.geometry?.dispose();
            const materials = Array.isArray(child.material) ? child.material : [child.material];
            materials.forEach(mat => mat?.dispose());
          }
        });
      }
      disposables.forEach(geometry => geometry.dispose());
      renderer.dispose();
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
      loadedObjectRef.current = null;
    };
  }, [base64, url, format]);

  const handleDownload = () => {
    const fmt = (format || guessMeshFormat(url) || 'glb').toLowerCase();
    const link = document.createElement('a');
    if (base64) {
      const buffer = base64ToArrayBuffer(base64);
      const blob = new Blob([buffer], { type: MIME_BY_FORMAT[fmt] || 'application/octet-stream' });
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      link.download = `mesh.${fmt}`;
      link.click();
      URL.revokeObjectURL(objectUrl);
    } else if (url) {
      link.href = url;
      link.download = `mesh.${fmt}`;
      link.click();
    }
  };

  return (
    <div className="mesh3d-viewer">
      <div className="mesh3d-toolbar">
        <button
          type="button"
          className={`mesh3d-tool-btn ${wireframe ? 'active' : ''}`}
          onClick={() => setWireframe(current => !current)}
          title="Toggle wireframe"
        >
          <Layers size={14} /> Wireframe
        </button>
        <button type="button" className="mesh3d-tool-btn" onClick={handleDownload} title="Download mesh">
          <Download size={14} /> Download
        </button>
      </div>
      <div className="mesh3d-canvas" ref={containerRef} />
      {error && <div className="mesh3d-error">{error}</div>}
    </div>
  );
}
