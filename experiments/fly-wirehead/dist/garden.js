import * as THREE from 'three';

const random = (n) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

// A garden at insect scale, kept outside the clear space around the platform.
export function createGarden(parent) {
  const garden = new THREE.Group(); parent.add(garden);
  const matte = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: .94, metalness: 0, flatShading: true, ...extra });
  const transform = new THREE.Object3D();
  function batch(geometry, material, items) {
    const mesh = new THREE.InstancedMesh(geometry, material, items.length);
    items.forEach((item, i) => {
      transform.position.set(...item.position);
      transform.rotation.set(...(item.rotation || [0, 0, 0]));
      transform.scale.set(...item.scale);
      transform.updateMatrix(); mesh.setMatrixAt(i, transform.matrix);
      if (item.color !== undefined) mesh.setColorAt(i, new THREE.Color(item.color));
    });
    mesh.receiveShadow = true; garden.add(mesh); return mesh;
  }
  const stone = new THREE.IcosahedronGeometry(1, 2);
  batch(stone, matte(0xffffff), [
    { position: [-13, -3.4, -17], scale: [15, 5, 9], color: 0x8cac65 },
    { position: [9, -3.9, -21], scale: [18, 6.5, 10], color: 0x9dbb79 },
    { position: [0, -4.5, -31], scale: [26, 8, 11], color: 0xb4ca90 },
  ]);
  batch(stone, matte(0xffffff), Array.from({ length: 32 }, (_, i) => {
    const side = i % 2 ? 1 : -1, size = .28 + random(i + 31) * .9;
    return { position: [side * (6 + random(i + 82) * 10), -1, random(i + 174) * 20 - 12], scale: [size * 1.6, size * .48, size], color: [0xbac59b, 0xd0cfac, 0xa8b583][i % 3] };
  }));

  // Folded, pointed blades catch sunlight along their central ridge.
  const rows = [[0, 0], [.22, .28], [.55, .39], [.85, .27], [1.15, 0]];
  const vertices = [];
  for (let i = 0; i < rows.length - 1; i++) {
    const [y, width] = rows[i], [nextY, nextWidth] = rows[i + 1];
    const ridge = Math.sin(y / 1.15 * Math.PI) * .12, nextRidge = Math.sin(nextY / 1.15 * Math.PI) * .12;
    for (const side of [-1, 1]) {
      const a = [0, y, ridge], b = [width * side, y, 0], c = [nextWidth * side, nextY, 0], d = [0, nextY, nextRidge];
      vertices.push(...a, ...b, ...c, ...a, ...c, ...d);
    }
  }
  const leafGeometry = new THREE.BufferGeometry();
  leafGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)); leafGeometry.computeVertexNormals();
  const leafColors = [0xaed585, 0xcce3a0, 0xbcdca0, 0xd8e8ae, 0xa6d19c];
  const leaves = [], stems = [];
  const plants = [[-7.8, -3.8, 5.4], [-5.8, -6, 6.4], [-2.8, -8, 4.2], [.9, -9, 5.7], [4.1, -8, 6.5], [7.1, -5, 5.3], [9.2, -.6, 4.3]];
  for (const [index, [x, z, height]] of plants.entries()) {
    stems.push({ position: [x, height / 2 - .9, z], scale: [.065, height, .065] });
    for (let j = 0; j < 7; j++) {
      const angle = j * 2.4 + index, size = 1.6 + random(index * 17 + j) * 1.3;
      leaves.push({ position: [x, -.7 + height * (.15 + j * .105), z], rotation: [.65 + random(j + index) * .35, angle, -.25], scale: [size, size, size], color: leafColors[(index + j) % leafColors.length] });
    }
  }
  for (let i = 0; i < 160; i++) {
    const x = random(i + 400) * 33 - 16.5, z = random(i + 650) * 24 - 17;
    if (Math.abs(x) < 5.9 && z > -4.4 && z < 4.4) continue;
    const size = .4 + random(i + 800) * 1.15;
    leaves.push({ position: [x, -.87, z], rotation: [random(i + 91) * .6, random(i + 77) * Math.PI * 2, .3], scale: [size * .48, size, size], color: leafColors[i % leafColors.length] });
  }
  const foliage = batch(leafGeometry, matte(0xffffff, { side: THREE.DoubleSide, emissive: 0xafd78d, emissiveIntensity: .22 }), leaves);

  // Daisies and soft pink flowers form an open border behind the subject.
  const petals = [], centers = [];
  const flowers = [[-6.6, -3.8, 3.2, 1], [-4.2, -6.2, 4.4, 1.1], [-.9, -7.4, 2.8, .7], [2.7, -7.1, 3.7, .9], [5.8, -4.6, 4.1, 1.15], [7.4, -.8, 2.4, .8], [-7, 2, 1.4, .7], [8, 3, 1.2, .6]];
  flowers.forEach(([x, z, height, size], index) => {
    stems.push({ position: [x, height / 2 - .9, z], scale: [.035, height, .035] });
    const y = height - .9;
    centers.push({ position: [x, y + .05, z], scale: [.27 * size, .15 * size, .27 * size] });
    for (let i = 0; i < 10; i++) {
      const angle = i * Math.PI / 5;
      petals.push({ position: [x + Math.cos(angle) * .47 * size, y, z + Math.sin(angle) * .47 * size], rotation: [0, -angle, 0], scale: [.46 * size, .075 * size, .18 * size], color: index % 3 === 1 ? 0xf1b3ba : 0xfff4ce });
    }
  });
  batch(new THREE.CylinderGeometry(1, 1, 1, 5), matte(0xb8d28f, { emissive: 0x9cbc7e, emissiveIntensity: .12 }), stems);
  const blossoms = batch(stone, matte(0xffffff), petals);
  batch(stone, matte(0xe9af36), centers);

  return {
    animate(time) {
      foliage.rotation.z = Math.sin(time * .42) * .006;
      blossoms.position.y = Math.sin(time * .8) * .014;
    },
  };
}
