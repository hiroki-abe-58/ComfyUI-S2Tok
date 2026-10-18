# S2Tok for ComfyUI（日本語概要）

> **非商用利用に限ります。** S2Tok のコードとモデルの重みは CC BY-NC 4.0 です（元になったモデルやデータも同様です）。本ノードパックは
> それらを実行するため、非商用目的でのみ使えます。詳しくは [README.md](README.md) の Licenses を参照してください。

動画のフレームから静的なシーンを **3D ガウシアン** として再構成する公式 **S2Tok**
（[Applied-Intuition-Open-Source/S2Tok](https://github.com/Applied-Intuition-Open-Source/S2Tok)、論文
[arXiv:2610.08978](https://arxiv.org/abs/2610.08978)）を ComfyUI で使うノードパックです。カメラ姿勢は不要で、1 フレームずつ再構成を更新
します。出力は ComfyUI 標準の `SPLAT` 型なので、コアノードでレンダリング（Render Splat）、`.ply` 保存（Create 3D File + Save Splat）、
プレビューができます。ノードは ComfyUI のプロセス内で動き、何もダウンロードしません。

> **非公式**の統合です。S2Tok の著者（Applied Intuition）、ETH Zurich（ZipSplat）、ByteDance（Depth Anything 3）、Meta（DINOv2）、
> Comfy Org とは関係がありません。

## ノード

| ノード | 役割 |
| --- | --- |
| **S2Tok Loader** | `models/s2tok` のチェックポイント、デバイス、常駐の有無を指定 |
| **S2Tok Reconstruct** | フレーム列（`IMAGE`）→ `SPLAT`（最後のフレームまでのシーン、または全フレーム分）と JSON レポート |
| **S2Tok Unload** | 常駐モデルを解放 |

## 検証済みの範囲（Windows 11 / RTX 5090 / ComfyUI v0.38.0 / torch 2.14.1）

- 2 つの動画シーン（41 / 46 フレーム）で、実キュー（Load Video → S2Tok Reconstruct → Create 3D File → Save Splat）が保存した `.ply` が、
  同じフレームから公式 `demo.py` が書き出す `scene.ply` と**バイト単位で一致**しました。
- 公式の `--precision bf16` は Windows 版 PyTorch（FlashAttention カーネルなし）で `No available kernel` になります。本パックは上流に
  提案した 1 文の修正（[S2Tok PR #1](https://github.com/Applied-Intuition-Open-Source/S2Tok/pull/1)）を同梱しており、bf16 の結果は
  修正後の公式コードとバイト一致、fp32 には影響しません。
- CI（GitHub Actions、Ubuntu / Windows、Python 3.10 / 3.12）では小さなランダム S2Tok で CPU テストと ComfyUI 実キューを実行し、同梱
  コードが固定コミットと一致することも検査しています。

## インストール

1. ComfyUI（V3 ノード API とスプラット系ノードを含む v0.38.0 で確認）、Python 3.10 以上、空き VRAM 約 8 GB の NVIDIA GPU（CPU でも約
   12 倍の時間で動きます）。
2. このフォルダを `ComfyUI/custom_nodes/` に置き、`python -m pip install -r custom_nodes/ComfyUI-S2Tok/requirements.txt`。
3. **非商用利用の場合のみ**、チェックポイント（6.9 GB、CC BY-NC 4.0）を取得して SHA-256 を検証:
   `python custom_nodes/ComfyUI-S2Tok/tools/download_model.py --models-dir ComfyUI/models/s2tok`
4. ComfyUI を再起動し、`workflows/gui/` のワークフローを読み込み、Load Video で自分の動画を選びます。

## 使い方の要点

- 静的なシーンの中や周りを移動する動画を使います。フレームは `demo.py` と同じく中央を正方形に切り出し 252 x 252 に縮小します。
  `frame_interval` で n フレームおきに使います（ワークフローは 4）。
- 長い動画は Load Video が全フレームを展開するため（1080p で 1 フレーム約 25 MB）、Video Frame Sample や Video Slice で先に間引いて
  ください。ComfyUI v0.38 の Video Frame Sample は WebM / MKV でフレーム数の数え間違いにより失敗します（修正案
  [ComfyUI PR #17154](https://github.com/Comfy-Org/ComfyUI/pull/17154)）。MP4 は問題ありません。
- 出力のガウシアンは最初のフレームのカメラ座標系（x 右、y 下、z 前、ComfyUI のスプラットノードと同じ規約）で、全体のスケールは
  不定です。最初のカメラから見るには Create Camera Info を quaternion モード、位置 0, 0, 0、クォータニオン 0, 0, 0, 1 にします。
- `bf16` は 41 フレームで 5.5 秒 → 3.3 秒、ガウシアン数は fp32 より 0.4〜0.7 % 少なくなりました（CUDA のみ）。

## ライセンス

- S2Tok のコード（`s2tok_comfy/vendor/`）とモデルの重み：**CC BY-NC 4.0、非商用のみ**。
- それ以外の本リポジトリのコード：MIT（[LICENSE](LICENSE) に適用範囲を明記）。ただしノードを使うと CC BY-NC のコードと重みを実行します。
- `docs/images`：CC BY-SA 4.0（CC BY-SA 4.0 の動画からの再構成のレンダリング、[出典](docs/images/README.md)）。
- 本パックは Comfy Registry には登録していません。
