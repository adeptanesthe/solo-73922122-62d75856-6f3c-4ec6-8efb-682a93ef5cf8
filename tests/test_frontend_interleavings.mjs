/* 前端判断提交与求解交接的关键交错测试（Node 运行，无第三方依赖）。
 *
 * 用 vm 加载真实的 app/static/app.js，DOM 与 fetch 全部打桩：
 * - fetch 挂起每个请求，由测试手动决定响应顺序，模拟网络乱序/延迟；
 * - 覆盖任务书要求的交错：
 *   1) 快速对同一对框先“同只”再“异只”：请求必须按操作顺序串行发出，
 *      服务端与页面都以最后一次保存为准；
 *   2) 求解中的旧任务在“异只”保存未确认时完成：不得展示旧计数；
 *   3) 保存失败：提示用户并从服务端重新同步，页面与服务端一致，
 *      且后续排队保存被取消；
 *   4) 求解按钮：先等待在途保存完成，再启动求解；
 *   5) 参数修改同样立即撤下旧计数、保存未确认时不展示旧任务结果；
 *   6) 会话切换后，旧会话在途保存的迟到响应不得污染新会话。
 *
 * 运行：node tests/test_frontend_interleavings.mjs
 */

import { readFileSync } from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const SRC = readFileSync(new URL("../app/static/app.js", import.meta.url), "utf8");

/* ------------------------------------------------------------ 打桩 */

function makeEl(id) {
  return {
    id,
    children: [],
    style: {},
    classList: { add() {}, remove() {} },
    textContent: "",
    innerHTML: "",
    className: "",
    disabled: false,
    value: "",
    files: [],
    addEventListener() {},
    appendChild(c) { this.children.push(c); },
    append(...cs) { this.children.push(...cs); },
    getContext() { return {}; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
}

class FakeImage {
  constructor() {
    this.complete = false;
    this.naturalWidth = 0;
    this.naturalHeight = 0;
  }
}

function makeFetch() {
  const requests = [];
  const fetch = (path, options = {}) =>
    new Promise((resolve, reject) => {
      requests.push({
        path,
        method: options.method || "GET",
        body: options.body ? JSON.parse(options.body) : null,
        respond(status, body) {
          resolve({
            ok: status >= 200 && status < 300,
            status,
            json: async () => body,
          });
        },
        fail(err) { reject(err); },
      });
    });
  return { fetch, requests };
}

function loadApp() {
  const els = new Map();
  const document = {
    getElementById(id) {
      if (!els.has(id)) els.set(id, makeEl(id));
      return els.get(id);
    },
    createElement() { return makeEl("dyn"); },
    addEventListener() {},
  };
  const alerts = [];
  const { fetch, requests } = makeFetch();
  const sandbox = {
    console,
    document,
    fetch,
    Image: FakeImage,
    alert: (m) => alerts.push(String(m)),
    setTimeout,
    clearTimeout,
    FormData: class { append() {} },
  };
  vm.createContext(sandbox);
  const exportSrc =
    "\n;globalThis.__app = { state, loadSession, setJudgment, removeJudgment," +
    " applyParams, solve, pollJob, findJudgment };";
  vm.runInContext(SRC + exportSrc, sandbox, { filename: "app.js" });
  return { app: sandbox.__app, alerts, requests, els };
}

/* ------------------------------------------------------------ 工具 */

const tick = async (n = 5) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const DETS = [
  { id: "a", frame: 0, rect: [0, 0, 8, 8], vector: [1, 0] },
  { id: "c", frame: 1, rect: [10, 0, 8, 8], vector: [1, 0] },
];
const PARAMS = { w_dist: 1, w_app: 10000, penalty: 15000 };

function payload(over = {}) {
  return {
    session_id: "s1",
    version: 1,
    n_frames: 2,
    detections: DETS,
    judgments: [],
    params: { ...PARAMS },
    ...over,
  };
}

function result(count) {
  return {
    feasible: true,
    message: "求解成功（全局最优）。",
    animal_count: count,
    n_trajectories: count,
    total_cost: 1.5,
    assoc_cost: 0.5,
    penalty_cost: 1.0,
    trajectories: [{ boxes: [{ id: "a", frame: 0 }], length: 1 }],
  };
}

function lastReq(requests, pred) {
  const found = requests.filter(pred);
  assert.ok(found.length > 0, "缺少预期请求");
  return found[found.length - 1];
}
const isPutJudgments = (r) => r.method === "PUT" && r.path.includes("/judgments");
const isSolve = (r) => r.method === "POST" && r.path.includes("/solve");
const isJobPoll = (r) => r.method === "GET" && r.path.includes("/api/jobs/");

/** 启动一次求解（不等待其返回），返回 solve() 的 Promise。 */
async function startSolve(app, requests, version) {
  const p = app.solve();
  await tick();
  lastReq(requests, isSolve).respond(200, { job_id: `job-${requests.length}`, version });
  await tick();
  return p;
}

/** 让一次求解完成并展示结果（版本 version、计数 count）。 */
async function displayResult(app, requests, version, count) {
  const p = startSolve(app, requests, version);
  await tick();
  lastReq(requests, isJobPoll).respond(200, { status: "done", version, result: result(count) });
  await tick();
  await p;
  assert.equal(app.state.result.animal_count, count, "结果应已展示");
}

/* ------------------------------------------------------------ 用例 */

async function testRapidRejudgmentSerialized() {
  const { app, requests } = loadApp();
  app.loadSession(payload());
  app.state.selection = ["a", "c"];

  app.setJudgment("same");
  assert.equal(app.state.judgments[0].type, "same", "点击后应立即乐观更新");
  await tick();
  assert.equal(requests.filter(isPutJudgments).length, 1);

  app.setJudgment("different"); // 快速改判：不得并发发出第二个请求
  assert.equal(app.state.judgments[0].type, "different", "第二次点击立即生效于页面");
  await tick();
  assert.equal(
    requests.filter(isPutJudgments).length, 1,
    "保存必须串行：前一个响应未返回前不得发出下一个",
  );

  // 服务端收到的顺序即用户操作顺序：先 same 后 different
  const put1 = lastReq(requests, isPutJudgments);
  assert.equal(put1.body.judgments[0].type, "same");
  put1.respond(200, { version: 2, judgments: [{ a: "a", b: "c", type: "same" }] });
  await tick();

  const puts = requests.filter(isPutJudgments);
  assert.equal(puts.length, 2, "第一个保存完成后才发出第二个");
  assert.equal(puts[1].body.judgments[0].type, "different");
  puts[1].respond(200, { version: 3, judgments: [{ a: "a", b: "c", type: "different" }] });
  await tick();

  assert.equal(app.state.version, 3);
  assert.equal(app.state.judgments[0].type, "different", "页面以最后一次保存为准");
  assert.equal(app.state.pendingSaves, 0);
  console.log("✓ 快速连续改判：请求按操作顺序串行保存，页面/服务端以最后一次为准");
}

async function testStaleJobHiddenWhileSavePending() {
  const { app, requests, els } = loadApp();
  app.loadSession(payload());
  await displayResult(app, requests, 1, 1); // 旧计数 1 已展示

  // 旧任务再次启动（基于 v1），随后用户立即改判
  await startSolve(app, requests, 1);

  app.state.selection = ["a", "c"];
  app.setJudgment("different"); // 修改一发起就撤下旧计数
  assert.equal(app.state.result, null, "修改发起即撤下旧计数");
  assert.equal(app.state.pendingSaves, 1);
  await tick();

  // 旧任务在“异只”保存未确认时完成 → 不得展示旧计数
  lastReq(requests, isJobPoll).respond(200, { status: "done", version: 1, result: result(1) });
  await tick();
  assert.equal(app.state.result, null, "保存未确认时不得展示旧任务结果");

  // 保存完成（版本推进到 v2）后，迟到的 v1 结果必须作废
  lastReq(requests, isPutJudgments).respond(
    200, { version: 2, judgments: [{ a: "a", b: "c", type: "different" }] });
  await tick();
  assert.equal(app.state.pendingSaves, 0);
  await sleep(250); // 等待延迟的轮询再次发起
  lastReq(requests, isJobPoll).respond(200, { status: "done", version: 1, result: result(1) });
  await tick();
  assert.equal(app.state.result, null, "迟到的旧版本结果不得展示");
  assert.match(els.get("solve-status").textContent, /作废/);
  console.log("✓ 保存未确认时旧任务结果不展示，迟到结果按版本作废");
}

async function testFailureAlertAndResync() {
  const { app, requests, alerts } = loadApp();
  app.loadSession(payload());
  app.state.selection = ["a", "c"];

  app.setJudgment("same");
  await tick();
  app.setJudgment("different"); // 排队中，将因前一个失败而被取消
  await tick();
  lastReq(requests, isPutJudgments).respond(400, { detail: "判断引用了不存在的框：a / ghost" });
  await tick();

  assert.equal(alerts.length, 1, "保存失败必须提示");
  assert.match(alerts[0], /保存判断失败/);

  const getSession = lastReq(
    requests, (r) => r.method === "GET" && r.path.includes("/api/sessions/"));
  getSession.respond(200, payload()); // 服务端仍是空判断
  await tick();

  assert.equal(app.state.judgments.length, 0, "失败后页面回滚到与服务端一致");
  assert.equal(app.state.version, 1);
  assert.equal(
    requests.filter(isPutJudgments).length, 1,
    "失败后排队中的保存被取消，不得再发出",
  );
  assert.equal(app.state.pendingSaves, 0);
  console.log("✓ 保存失败：提示 + 取消排队保存 + 页面与服务端重新同步");
}

async function testSolveWaitsForPendingSaves() {
  const { app, requests } = loadApp();
  app.loadSession(payload());
  app.state.selection = ["a", "c"];
  app.setJudgment("different");
  await tick();

  const solving = app.solve(); // 不 await：应先等待保存完成
  await tick();
  assert.equal(requests.filter(isSolve).length, 0, "保存未确认前不得启动求解");

  lastReq(requests, isPutJudgments).respond(
    200, { version: 2, judgments: [{ a: "a", b: "c", type: "different" }] });
  await tick();
  const solveReq = lastReq(requests, isSolve);
  solveReq.respond(200, { job_id: "j-new", version: 2 });
  await tick();
  await solving;
  assert.equal(app.state.jobId, "j-new");
  console.log("✓ 求解先等待在途保存完成，基于最后一次成功保存的输入启动");
}

async function testParamsChangeAlsoInvalidates() {
  const { app, requests } = loadApp();
  app.loadSession(payload());
  await displayResult(app, requests, 1, 1);

  await startSolve(app, requests, 1);

  app.applyParams(); // 参数修改一发起就撤下旧计数
  assert.equal(app.state.result, null, "参数修改一发起就撤下旧计数");
  await tick();
  lastReq(requests, isJobPoll).respond(200, { status: "done", version: 1, result: result(1) });
  await tick();
  assert.equal(app.state.result, null, "参数保存未确认时不得展示旧任务结果");

  lastReq(requests, (r) => r.method === "PUT" && r.path.includes("/params"))
    .respond(200, { version: 2, params: PARAMS });
  await tick();
  await sleep(250);
  lastReq(requests, isJobPoll).respond(200, { status: "done", version: 1, result: result(1) });
  await tick();
  assert.equal(app.state.result, null, "迟到的旧版本结果不得因参数流程而复活");
  console.log("✓ 参数修改同样立即撤下旧计数，旧任务结果不得复活");
}

async function testSessionSwitchIgnoresLateSave() {
  const { app, requests } = loadApp();
  app.loadSession(payload());
  app.state.selection = ["a", "c"];
  app.setJudgment("same"); // s1 的保存在途
  await tick();

  app.loadSession(payload({ session_id: "s2", version: 1 })); // 切换会话
  assert.equal(app.state.sessionId, "s2");

  lastReq(requests, isPutJudgments).respond(
    200, { version: 9, judgments: [{ a: "a", b: "c", type: "same" }] });
  await tick();
  assert.equal(app.state.version, 1, "旧会话的迟到响应不得污染新会话版本");
  assert.equal(app.state.judgments.length, 0, "旧会话的迟到响应不得污染新会话判断");
  assert.equal(app.state.pendingSaves, 0, "在途计数保持平衡");
  console.log("✓ 会话切换后旧会话在途保存的迟到响应被忽略");
}

/* ------------------------------------------------------------ 运行 */

const tests = [
  testRapidRejudgmentSerialized,
  testStaleJobHiddenWhileSavePending,
  testFailureAlertAndResync,
  testSolveWaitsForPendingSaves,
  testParamsChangeAlsoInvalidates,
  testSessionSwitchIgnoresLateSave,
];

for (const t of tests) {
  await t();
}
console.log(`\n全部 ${tests.length} 个前端交错测试通过。`);
