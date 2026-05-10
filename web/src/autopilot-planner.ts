import { latLngToMercator, mercatorToLatLng } from "./utils";

type BBoxMercator = [number, number, number, number];
type Split = "train" | "val" | "test";

export type AutopilotPlanInput = {
  targetPositiveCount: number;
  sceneSizeM?: number;
  tileSizeM?: number;
  imagePx?: number;
  maxPanels?: number;
  perimeterBudget?: number;
};

export type AutopilotCoarseCellStatus = "background" | "candidate" | "urban";

export type AutopilotCoarseCell = {
  id: string;
  row: number;
  col: number;
  status: AutopilotCoarseCellStatus;
  urbanScore: number;
  bboxMercator: BBoxMercator;
  panelId?: string;
};

export type AutopilotBvhCell = {
  id: string;
  depth: number;
  layerAboveBase?: number;
  sizeM?: number;
  status: AutopilotCoarseCellStatus;
  urbanScore: number;
  maxUrbanScore?: number;
  urbanRatio?: number;
  bboxMercator: BBoxMercator;
  panelId?: string | null;
};

export type AutopilotPanel = {
  id: string;
  name: string;
  split: Split;
  rank: number;
  coarseCellCount: number;
  urbanScore: number;
  bboxMercator: BBoxMercator;
  plannedScenes: number;
  estimatedPositiveCount: number;
};

export type AutopilotCellStatus = "panel" | "frontier" | "selected";

export type AutopilotCell = {
  id: string;
  panelId: string;
  panelName: string;
  status: AutopilotCellStatus;
  depth: number;
  rank: number;
  score: number;
  bboxMercator: BBoxMercator;
  center: {
    latitude: number;
    longitude: number;
  };
  sizeM: number;
  sceneId?: string;
};

export type AutopilotScene = {
  scene_id: string;
  city: string;
  split: Split;
  latitude: number;
  longitude: number;
  size_m: number;
  image_px: number;
  autopilot_rank: number;
  autopilot_score: number;
  autopilot_city_id: string;
  autopilot_cell_id: string;
};

export type AutopilotPlan = {
  version: 2 | 3 | 4 | 5 | 6;
  mode: "swiss-lowres-urban-grid";
  source?: string;
  segmentation?: {
    sourceLayer: string;
    method: string;
    surfaceThreshold: number;
    surfaceCoverage: number;
  };
  gridGeometry?: {
    baseGridSizeM: number;
    sceneGridSizeM: number;
    sceneLayerAboveBase: number;
    originMercator: [number, number];
    alignment?: string;
    rule: string;
  };
  targetPositiveCount: number;
  estimatedPositiveCount: number;
  estimatedPositivePerScene: number;
  sceneSizeM: number;
  tileSizeM: number;
  imagePx: number;
  maxPanels: number;
  sceneBudget: number;
  coarseGrid: {
    rows: number;
    cols: number;
    bboxMercator: BBoxMercator;
  };
  coarseCells: AutopilotCoarseCell[];
  bvhCells?: AutopilotBvhCell[];
  panels: AutopilotPanel[];
  cells: AutopilotCell[];
  scenes: AutopilotScene[];
  createdAt: string;
};

const DEFAULT_SCENE_SIZE_M = 800;
const DEFAULT_TILE_SIZE_M = 25;
const DEFAULT_IMAGE_PX = 2048;
const ESTIMATED_POSITIVE_PER_SCENE = 7;
const DEFAULT_MAX_PANELS = 8;
const MAX_SCENE_BUDGET = 120;
const COARSE_ROWS = 40;
const COARSE_COLS = 64;

const SWISS_SCAN_BOUNDS = {
  south: 45.78,
  west: 5.86,
  north: 47.84,
  east: 10.55,
};

function bboxWidth([minX, , maxX]: BBoxMercator) {
  return maxX - minX;
}

function bboxHeight([, minY, , maxY]: BBoxMercator) {
  return maxY - minY;
}

function bboxCenter(bbox: BBoxMercator) {
  const [latitude, longitude] = mercatorToLatLng((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2);
  return { latitude, longitude };
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function stableNoise(x: number, y: number) {
  const value = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function gaussian(value: number, center: number, spread: number) {
  const d = (value - center) / spread;
  return Math.exp(-d * d);
}

function corridorScore(x: number, y: number) {
  const centralBand = gaussian(y, 0.5 + Math.sin(x * Math.PI * 2) * 0.08, 0.16);
  const plateauBand = gaussian(y, 0.72 - x * 0.12, 0.09);
  const alpinePenalty = gaussian(y, 0.16, 0.18) * 0.42;
  return clamp01(centralBand * 0.58 + plateauBand * 0.24 - alpinePenalty);
}

function lowResolutionUrbanScore(row: number, col: number) {
  const x = (col + 0.5) / COARSE_COLS;
  const y = (row + 0.5) / COARSE_ROWS;
  const texture = stableNoise(Math.floor(x * 10), Math.floor(y * 8));
  const block = stableNoise(Math.floor(x * 18), Math.floor(y * 14));
  const repeatedEdges = (Math.sin((x + block * 0.2) * Math.PI * 17) + Math.cos((y + texture * 0.15) * Math.PI * 13) + 2) / 4;
  return clamp01(corridorScore(x, y) * 0.72 + repeatedEdges * 0.22 + texture * 0.18);
}

function swissGridBBox(): BBoxMercator {
  const southWest = latLngToMercator(SWISS_SCAN_BOUNDS.south, SWISS_SCAN_BOUNDS.west);
  const northEast = latLngToMercator(SWISS_SCAN_BOUNDS.north, SWISS_SCAN_BOUNDS.east);
  return [southWest.x, southWest.y, northEast.x, northEast.y];
}

function cellBBox(gridBBox: BBoxMercator, row: number, col: number, rows: number, cols: number): BBoxMercator {
  const [minX, minY, maxX, maxY] = gridBBox;
  const width = (maxX - minX) / cols;
  const height = (maxY - minY) / rows;
  const left = minX + col * width;
  const right = left + width;
  const top = maxY - row * height;
  const bottom = top - height;
  return [left, bottom, right, top];
}

function buildCoarseCells(gridBBox: BBoxMercator) {
  const cells: AutopilotCoarseCell[] = [];
  for (let row = 0; row < COARSE_ROWS; row += 1) {
    for (let col = 0; col < COARSE_COLS; col += 1) {
      const urbanScore = lowResolutionUrbanScore(row, col);
      cells.push({
        id: `ch-r${String(row).padStart(2, "0")}-c${String(col).padStart(2, "0")}`,
        row,
        col,
        status: urbanScore >= 0.58 ? "urban" : urbanScore >= 0.45 ? "candidate" : "background",
        urbanScore,
        bboxMercator: cellBBox(gridBBox, row, col, COARSE_ROWS, COARSE_COLS),
      });
    }
  }
  return cells;
}

function cellKey(row: number, col: number) {
  return `${row}:${col}`;
}

function connectedUrbanClusters(cells: AutopilotCoarseCell[]) {
  const byKey = new Map(cells.map((cell) => [cellKey(cell.row, cell.col), cell]));
  const visited = new Set<string>();
  const clusters: AutopilotCoarseCell[][] = [];

  for (const cell of cells) {
    if (cell.status !== "urban" || visited.has(cell.id)) continue;
    const cluster: AutopilotCoarseCell[] = [];
    const queue = [cell];
    visited.add(cell.id);

    while (queue.length) {
      const current = queue.shift()!;
      cluster.push(current);
      for (const [row, col] of [
        [current.row - 1, current.col],
        [current.row + 1, current.col],
        [current.row, current.col - 1],
        [current.row, current.col + 1],
      ]) {
        const next = byKey.get(cellKey(row, col));
        if (!next || next.status !== "urban" || visited.has(next.id)) continue;
        visited.add(next.id);
        queue.push(next);
      }
    }

    clusters.push(cluster);
  }

  return clusters;
}

function mergeBBoxes(bboxes: BBoxMercator[]): BBoxMercator {
  return [
    Math.min(...bboxes.map((bbox) => bbox[0])),
    Math.min(...bboxes.map((bbox) => bbox[1])),
    Math.max(...bboxes.map((bbox) => bbox[2])),
    Math.max(...bboxes.map((bbox) => bbox[3])),
  ];
}

function splitForRank(rank: number): Split {
  if (rank % 10 === 0) return "test";
  if (rank % 5 === 0) return "val";
  return "train";
}

function buildPanels(cells: AutopilotCoarseCell[], sceneBudget: number, maxPanels: number) {
  const clusters = connectedUrbanClusters(cells)
    .map((cluster) => {
      const urbanScore = cluster.reduce((sum, cell) => sum + cell.urbanScore, 0) / cluster.length;
      const bboxMercator = mergeBBoxes(cluster.map((cell) => cell.bboxMercator));
      return { cluster, urbanScore, bboxMercator, weight: urbanScore * Math.sqrt(cluster.length) };
    })
    .sort((left, right) => right.weight - left.weight);

  const activeClusters = clusters.slice(0, maxPanels);
  const totalWeight = activeClusters.reduce((sum, cluster) => sum + cluster.weight, 0) || 1;
  let remainingScenes = sceneBudget;

  const panels = clusters.map((entry, index) => {
    const isActive = index < maxPanels;
    const plannedScenes = isActive ? Math.max(1, Math.floor((entry.weight / totalWeight) * sceneBudget)) : 0;
    remainingScenes -= plannedScenes;
    return {
      id: `urban-panel-${String(index + 1).padStart(3, "0")}`,
      name: `Urban panel ${String(index + 1).padStart(3, "0")}`,
      split: splitForRank(index + 1),
      rank: index + 1,
      coarseCellCount: entry.cluster.length,
      urbanScore: entry.urbanScore,
      bboxMercator: entry.bboxMercator,
      plannedScenes,
      estimatedPositiveCount: plannedScenes * ESTIMATED_POSITIVE_PER_SCENE,
    } satisfies AutopilotPanel;
  });

  for (const panel of panels.slice(0, maxPanels)) {
    if (remainingScenes <= 0) break;
    panel.plannedScenes += 1;
    panel.estimatedPositiveCount += ESTIMATED_POSITIVE_PER_SCENE;
    remainingScenes -= 1;
  }

  const clusterCells = clusters.flatMap((entry, index) => entry.cluster.map((cell) => ({ cell, panelId: panels[index]?.id })));
  return {
    panels,
    coarseCells: cells.map((cell) => {
      const match = clusterCells.find((entry) => entry.cell.id === cell.id);
      return match?.panelId ? { ...cell, panelId: match.panelId } : cell;
    }),
  };
}

function scorePanelCell(panel: AutopilotPanel, bbox: BBoxMercator, depth: number) {
  const panelCenter = bboxCenter(panel.bboxMercator);
  const center = latLngToMercator(panelCenter.latitude, panelCenter.longitude);
  const cellCenterX = (bbox[0] + bbox[2]) / 2;
  const cellCenterY = (bbox[1] + bbox[3]) / 2;
  const panelScale = Math.max(bboxWidth(panel.bboxMercator), bboxHeight(panel.bboxMercator));
  const distanceBias = 1 / (1 + Math.hypot(cellCenterX - center.x, cellCenterY - center.y) / Math.max(1, panelScale * 0.44));
  const shapeBias = Math.min(bboxWidth(bbox), bboxHeight(bbox)) / Math.max(1, Math.max(bboxWidth(bbox), bboxHeight(bbox)));
  return panel.urbanScore * distanceBias * (0.8 + shapeBias * 0.2) * (1 / (1 + depth * 0.03));
}

function splitCell(panel: AutopilotPanel, cell: AutopilotCell): [AutopilotCell, AutopilotCell] {
  const [minX, minY, maxX, maxY] = cell.bboxMercator;
  const splitVertical = bboxWidth(cell.bboxMercator) >= bboxHeight(cell.bboxMercator);
  const middle = splitVertical ? (minX + maxX) / 2 : (minY + maxY) / 2;
  const firstBBox: BBoxMercator = splitVertical ? [minX, minY, middle, maxY] : [minX, minY, maxX, middle];
  const secondBBox: BBoxMercator = splitVertical ? [middle, minY, maxX, maxY] : [minX, middle, maxX, maxY];

  return [firstBBox, secondBBox].map((bbox, index) => {
    const width = bboxWidth(bbox);
    const height = bboxHeight(bbox);
    return {
      ...cell,
      id: `${cell.id}.${index + 1}`,
      status: "frontier",
      depth: cell.depth + 1,
      score: scorePanelCell(panel, bbox, cell.depth + 1),
      bboxMercator: bbox,
      center: bboxCenter(bbox),
      sizeM: Math.max(width, height),
    } satisfies AutopilotCell;
  }) as [AutopilotCell, AutopilotCell];
}

function rankCells(cells: AutopilotCell[]) {
  return cells
    .slice()
    .sort((left, right) => right.score - left.score || left.panelName.localeCompare(right.panelName) || left.id.localeCompare(right.id))
    .map((cell, index) => ({ ...cell, rank: index + 1 }));
}

function plannedCellsForPanel(panel: AutopilotPanel, sceneSizeM: number) {
  const root: AutopilotCell = {
    id: panel.id,
    panelId: panel.id,
    panelName: panel.name,
    status: "panel",
    depth: 0,
    rank: 0,
    score: scorePanelCell(panel, panel.bboxMercator, 0),
    bboxMercator: panel.bboxMercator,
    center: bboxCenter(panel.bboxMercator),
    sizeM: Math.max(bboxWidth(panel.bboxMercator), bboxHeight(panel.bboxMercator)),
  };

  let queue = [root];
  while (queue.length < panel.plannedScenes) {
    const next = queue.slice().sort((left, right) => right.score - left.score)[0];
    if (!next || next.sizeM <= sceneSizeM * 1.2) break;
    queue = queue.filter((cell) => cell.id !== next.id);
    queue.push(...splitCell(panel, next));
  }

  const selected = rankCells(queue).slice(0, panel.plannedScenes).map((cell) => ({ ...cell, status: "selected" as const }));
  const frontier = rankCells(queue)
    .slice(panel.plannedScenes, panel.plannedScenes + 3)
    .map((cell) => ({ ...cell, status: "frontier" as const }));

  return [root, ...selected, ...frontier];
}

export function buildAutopilotPlan(input: AutopilotPlanInput): AutopilotPlan {
  const targetPositiveCount = Math.max(20, Math.round(input.targetPositiveCount || 500));
  const sceneSizeM = Math.max(400, input.sceneSizeM ?? DEFAULT_SCENE_SIZE_M);
  const tileSizeM = Math.max(10, input.tileSizeM ?? DEFAULT_TILE_SIZE_M);
  const imagePx = Math.max(512, input.imagePx ?? DEFAULT_IMAGE_PX);
  const maxPanels = Math.max(1, Math.min(24, Math.round(input.maxPanels ?? DEFAULT_MAX_PANELS)));
  const requestedSceneBudget = input.perimeterBudget ?? Math.ceil(targetPositiveCount / ESTIMATED_POSITIVE_PER_SCENE);
  const sceneBudget = Math.min(MAX_SCENE_BUDGET, Math.max(4, Math.round(requestedSceneBudget)));
  const gridBBox = swissGridBBox();
  const initialCoarseCells = buildCoarseCells(gridBBox);
  const { panels, coarseCells } = buildPanels(initialCoarseCells, sceneBudget, maxPanels);
  const selectedCells = panels
    .slice(0, maxPanels)
    .flatMap((panel) => plannedCellsForPanel(panel, sceneSizeM))
    .filter((cell) => cell.status === "selected");
  const selected = rankCells(selectedCells).slice(0, sceneBudget);
  const sceneIds = new Map(selected.map((cell, index) => [cell.id, `auto-panel-${String(index + 1).padStart(3, "0")}`]));

  const cells = [
    ...panels.slice(0, maxPanels).map(
      (panel) =>
        ({
          id: panel.id,
          panelId: panel.id,
          panelName: panel.name,
          status: "panel" as const,
          depth: 0,
          rank: panel.rank,
          score: panel.urbanScore,
          bboxMercator: panel.bboxMercator,
          center: bboxCenter(panel.bboxMercator),
          sizeM: Math.max(bboxWidth(panel.bboxMercator), bboxHeight(panel.bboxMercator)),
        }) satisfies AutopilotCell,
    ),
    ...selected.map((cell) => ({
      ...cell,
      sceneId: sceneIds.get(cell.id),
    })),
  ];

  const scenes = selected.map((cell, index) => ({
    scene_id: sceneIds.get(cell.id) ?? `auto-panel-${String(index + 1).padStart(3, "0")}`,
    city: cell.panelName,
    split: panels.find((panel) => panel.id === cell.panelId)?.split ?? "train",
    latitude: cell.center.latitude,
    longitude: cell.center.longitude,
    size_m: sceneSizeM,
    image_px: imagePx,
    autopilot_rank: index + 1,
    autopilot_score: cell.score,
    autopilot_city_id: cell.panelId,
    autopilot_cell_id: cell.id,
  }));

  return {
    version: 2,
    mode: "swiss-lowres-urban-grid",
    targetPositiveCount,
    estimatedPositiveCount: scenes.length * ESTIMATED_POSITIVE_PER_SCENE,
    estimatedPositivePerScene: ESTIMATED_POSITIVE_PER_SCENE,
    sceneSizeM,
    tileSizeM,
    imagePx,
    maxPanels,
    sceneBudget,
    coarseGrid: {
      rows: COARSE_ROWS,
      cols: COARSE_COLS,
      bboxMercator: gridBBox,
    },
    coarseCells,
    panels,
    cells,
    scenes,
    createdAt: new Date().toISOString(),
  };
}
