---
title: "Dinorunner: Chrome's Dino Game as a Portable C Engine You Can Run Anywhere"
date: 2026-10-28 12:42:00 +0100
categories: [Embedded, Games]
tags: [c, cpp, rust, python, webassembly, wasm, game, embedded, portability, open-source]
toc: true
description: "A dependency-free C port of Chrome's offline dinosaur game — no stdlib, no heap, 1264 bytes of state — with C++, Rust and Python bindings, and a live WebAssembly build you can play right here in the browser."
image:
    path: img/dinorunner/cover.gif
    alt: Dinorunner running
---

Everyone has met Chrome's dinosaur: the little T-Rex that appears when your connection drops, hopping cacti and ducking pterodactyls. [Dinorunner](https://github.com/AKJ7/dinorunner) is a faithful port of that game, rewritten as a **tiny, dependency-free C engine** with bindings for C++, Rust and Python — and a build that compiles to WebAssembly and runs in the browser.

The point of the project is portability. The engine knows nothing about screens, keyboards, files or clocks. It computes game state and asks the host to do the rest through a handful of callbacks, which means the same core runs under SDL on a desktop, under pygame in Python, or — as below — on a `<canvas>` in your browser through a Rust/WebAssembly shim.

## Play it right here

The game below is the real engine: libdinorunner's C core, driven through the Rust binding and compiled to WebAssembly. It draws the original sprite sheet, plays the original sounds, keeps your high score in `localStorage`, and inverts to night mode just like the original.

<div style="margin: 1rem 0; border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; overflow: hidden;">
  <iframe src="/assets/dinorunner/embed.html"
          title="Dinorunner live WebAssembly demo"
          loading="lazy"
          style="width: 100%; height: 320px; border: 0; display: block;">
  </iframe>
</div>

> Click the game to give it focus, then <kbd>Space</kbd> or <kbd>&uarr;</kbd> to jump and <kbd>&darr;</kbd> to duck. On a phone, just tap. Sound unlocks after your first interaction, as browsers require.
{: .prompt-tip }

## Why bother porting a game that already exists?

Because the interesting part is *how small and portable* the thing can be. The engine is written in plain C with a deliberately austere set of constraints:

- **No hardware dependencies** — it runs anywhere a C compiler does.
- **No compiler-specific extensions** — compile and go.
- **No `stdlib`** — it needs only the freestanding `stdarg.h`.
- **No heap allocations** — nothing is ever `malloc`'d.
- **No `typedef`s**, and the whole game state fits in **1264 bytes** on x86-64.
- It still supports **night mode** and **variadic jump heights** (a short tap gives a lower jump than a held key).

That makes it a drop-in backend for anything from a desktop app to a microcontroller with a few kilobytes of RAM — anywhere you can draw a handful of sprites.

## How the engine talks to the outside world

Dinorunner never calls into the system directly. Instead, the host hands it a struct of function pointers at startup, and the engine calls back into them when it needs something:

```c
struct dinorunner_callbacks_s {
  unsigned long  (*gettimestamp)(void* user_data);
  unsigned char  (*clearcanvas)(void* user_data);
  unsigned char  (*draw)(enum dinorunner_sprite_e sprite, const struct pos_s* pos,
                         unsigned char opacity, void* user_data);
  unsigned char  (*playsound)(enum dinorunner_sound_e sound, void* user_data);
  unsigned char  (*vibrate)(unsigned duration, void* user_data);
  unsigned char  (*readhighscore)(unsigned long* high_score, void* user_data);
  unsigned char  (*writehighscore)(unsigned long high_score, void* user_data);
  unsigned char  (*log)(const char* message, void* user_data);
};
```

Only `gettimestamp`, `clearcanvas` and `draw` are required; the rest may be `NULL`. Each callback receives the `user_data` you passed to `dinorunner_init`, so several games with different hosts can run in the same program. The loop a host runs is tiny:

```c
struct dinorunner_s game;
dinorunner_init(&game, &dimension, &callbacks, user_data);
for (;;) {
    dinorunner_onkeydown(&game);   /* or onkeyup / onkeynone from your input */
    dinorunner_update(&game);      /* advances state and calls draw(...) */
}
```

Night mode is not something the engine paints — it simply reports it through `dinorunner_isinverted`, and the host decides how to render it. The WebAssembly demo above, for instance, just toggles a CSS `filter: invert(1)` on the canvas.

## One engine, four languages

The repository ships the core plus three language bindings, all wrapping the same C library:

| Binding | What it is |
|---|---|
| **libdinorunner** (C) | The engine itself — compile it in directly, or as a static/shared library via CMake. |
| **dinorunner-cpp** | A header-only C++17 wrapper. You subclass `dinorunner::Dinorunner` and implement its virtual `Draw`, `PlaySound`, `GetTimestamp`, … methods. |
| **dinorunner-rust** | A Rust binding whose build script compiles the C sources itself — no CMake step needed first. |
| **dinorunner-python** | Python bindings over the same C core. |

Pulling the engine into a CMake project is a `FetchContent` away:

```cmake
include(FetchContent)
FetchContent_Declare(
    DINORUNNER
    GIT_REPOSITORY https://github.com/AKJ7/dinorunner
    GIT_TAG master
    GIT_PROGRESS TRUE
    SOURCE_SUBDIR dinorunner
)
FetchContent_MakeAvailable(DINORUNNER)
add_executable(cpp-example main.cpp)
target_link_libraries(cpp-example PUBLIC dinorunner::dinorunner_static)
```

## How the browser demo works

The demo you played is the `demo/rust-webassembly` build. The Rust binding implements the engine's `Platform` trait for the browser, mapping each engine request onto a Web API:

| Engine request | Browser API |
|---|---|
| `timestamp` | `performance.now()` |
| `clear_canvas`, `draw` | Canvas 2D context, cutting sprites out of the sprite sheet |
| `play_sound` | `HTMLAudioElement` |
| `vibrate` | `navigator.vibrate()` |
| `read/write_high_score` | `localStorage` |
| `log` | `console.log()` |

The C sources are compiled straight to `wasm32-unknown-unknown` by the Rust build script (with clang, since gcc cannot emit WebAssembly), and `wasm-bindgen` generates the JavaScript glue. A `requestAnimationFrame` loop then calls `Dinorunner::update` once per displayed frame. The target has no C standard library — which is fine, because the engine never needed one. Building it yourself is a single script:

```shell
cd demo/rust-webassembly
./build.sh --serve      # builds into www/ and serves on http://localhost:8000
```

There are also an **SDL** demo in C and a **pygame** demo in Python under `demo/`, both driving the exact same engine.

## Try it

The whole thing is on [GitHub](https://github.com/AKJ7/dinorunner) — the C engine, the C++/Rust/Python bindings, and the SDL, pygame and WebAssembly demos. It is currently at version 2.0.0. If you want a small, self-contained game engine to embed somewhere unusual — a kiosk, a dashboard easter egg, a bare-metal board — clone it, wire up three callbacks, and you have a dino hopping cacti.

Assets come from the wonderful [t-rex-runner](https://github.com/wayou/t-rex-runner) project; issues and pull requests are welcome.
