fn main() {
    println!("cargo:rerun-if-env-changed=MESH_LLM_GPU_BENCH_RUST_ONLY");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rustc-check-cfg=cfg(mesh_llm_gpu_bench_has_ggml_probe)");
    if std::env::var_os("MESH_LLM_GPU_BENCH_RUST_ONLY").is_some() {
        return;
    }

    if target_os_is("macos") {
        build_metal();
    }

    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build_cuda();
    }

    if std::env::var_os("CARGO_FEATURE_HIP").is_some() {
        build_hip();
    }

    if std::env::var_os("CARGO_FEATURE_INTEL").is_some() {
        build_intel();
    }

    if std::env::var_os("CARGO_FEATURE_GGML_PROBE").is_some() {
        build_ggml_probe();
    }
}

fn target_os_is(os: &str) -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok(os)
}

fn build_metal() {
    let object = out_path("mesh_llm_gpu_bench_metal.o");
    run_or_panic({
        let mut command = std::process::Command::new("clang");
        command.arg("-O3").arg("-fobjc-arc").arg("-fPIC").arg("-c");
        add_macos_target_flags(&mut command);
        command
            .arg("native/metal/membench_metal.m")
            .arg("-o")
            .arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_metal");

    println!("cargo:rerun-if-changed=native/metal/membench_metal.m");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
}

fn add_macos_target_flags(command: &mut std::process::Command) {
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        Ok(arch) => panic!("unsupported macOS Metal benchmark target architecture: {arch}"),
        Err(err) => panic!("CARGO_CFG_TARGET_ARCH is required for Metal benchmark build: {err}"),
    };
    command.arg("-arch").arg(arch);

    if let Some(sdk_path) = macos_sdk_path() {
        command.arg("-isysroot").arg(sdk_path);
    }

    let deployment_target =
        std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "13.0".to_string());
    command.arg(format!("-mmacosx-version-min={deployment_target}"));
}

fn macos_sdk_path() -> Option<String> {
    let output = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (!path.is_empty()).then(|| path.to_string())
}

fn native_source(dir: &str, name: &str) -> String {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir
        .join("native")
        .join(dir)
        .join(name)
        .display()
        .to_string()
}

fn out_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(name)
}

fn write_wrapper(name: &str, source: &str, symbol: &str) -> std::path::PathBuf {
    let wrapper = out_path(name);
    let body = format!(
        "#define main {symbol}_program_main\n#include \"{source}\"\n#undef main\nextern \"C\" int {symbol}(void) {{ char arg0[] = \"{symbol}\"; char arg1[] = \"--json\"; char *argv[] = {{ arg0, arg1, nullptr }}; return {symbol}_program_main(2, argv); }}\n"
    );
    std::fs::write(&wrapper, body).unwrap();
    println!("cargo:rerun-if-changed={source}");
    wrapper
}

fn run_or_panic(mut command: std::process::Command) {
    let status = command.status().unwrap_or_else(|err| {
        panic!(
            "failed to run native benchmark compiler {:?}: {err}",
            command
        )
    });
    assert!(
        status.success(),
        "native benchmark compiler {:?} failed with {status}",
        command
    );
}

fn archive_static_lib(object: &std::path::Path, lib_name: &str) {
    if cfg!(windows) {
        cc::Build::new().object(object).compile(lib_name);
        return;
    }

    let lib_path = out_path(&format!("lib{lib_name}.a"));
    run_or_panic({
        let mut command = std::process::Command::new("ar");
        command.arg("crus").arg(&lib_path).arg(object);
        command
    });
    println!("cargo:rustc-link-search=native={}", out_path("").display());
    println!("cargo:rustc-link-lib=static={lib_name}");
}

fn target_is_windows_msvc() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
}

fn target_uses_static_crt() -> bool {
    std::env::var("CARGO_CFG_TARGET_FEATURE")
        .map(|features| features.split(',').any(|feature| feature == "crt-static"))
        .unwrap_or(false)
}

fn add_windows_cuda_crt_flags(command: &mut std::process::Command) {
    if target_is_windows_msvc() {
        let runtime = if target_uses_static_crt() {
            "/MT"
        } else {
            "/MD"
        };
        command.arg("-Xcompiler").arg(runtime);
    }
}

fn add_windows_hip_crt_flags(command: &mut std::process::Command) {
    if target_is_windows_msvc() {
        let runtime = if target_uses_static_crt() {
            "-fms-runtime-lib=static"
        } else {
            "-fms-runtime-lib=dll"
        };
        command.arg(runtime);
    }
}

fn build_cuda() {
    let source = native_source("cuda", "membench-fingerprint.cu");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_cuda_wrapper.cu",
        &source,
        "mesh_llm_gpu_bench_cuda_main",
    );
    let object = out_path("mesh_llm_gpu_bench_cuda.o");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(nvcc);
        command.arg("-O3").arg("-std=c++17");
        add_windows_cuda_crt_flags(&mut command);
        if !cfg!(windows) {
            command.arg("-Xcompiler").arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_cuda");
    for dir in cuda_library_search_dirs() {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    if !cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn cuda_library_search_dirs() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = std::env::var_os(key).map(std::path::PathBuf::from) {
            roots.push(root);
        }
    }
    if let Some(root) = cuda_root_from_nvcc() {
        roots.push(root);
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mut dirs = Vec::new();
    for root in roots {
        push_existing_cuda_lib_dir(&mut dirs, root.join("lib64"));
        push_existing_cuda_lib_dir(&mut dirs, root.join("lib"));
        if target_arch == "x86_64" {
            push_existing_cuda_lib_dir(&mut dirs, root.join("targets/x86_64-linux/lib"));
        }
        if target_arch == "aarch64" {
            push_existing_cuda_lib_dir(&mut dirs, root.join("targets/aarch64-linux/lib"));
        }
    }
    dirs
}

fn cuda_root_from_nvcc() -> Option<std::path::PathBuf> {
    let nvcc = std::env::var_os("NVCC")
        .map(std::path::PathBuf::from)
        .or_else(|| which_in_path("nvcc"))?;
    nvcc.parent()?.parent().map(std::path::Path::to_path_buf)
}

fn which_in_path(binary: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

fn push_existing_cuda_lib_dir(dirs: &mut Vec<std::path::PathBuf>, dir: std::path::PathBuf) {
    if dir.is_dir() && !dirs.iter().any(|existing| existing == &dir) {
        dirs.push(dir);
    }
}

fn build_hip() {
    let source = native_source("hip", "membench-fingerprint.hip");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_hip_wrapper.hip",
        &source,
        "mesh_llm_gpu_bench_hip_main",
    );
    let object = out_path("mesh_llm_gpu_bench_hip.o");
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(hipcc);
        command.arg("-O3").arg("-std=c++17");
        add_windows_hip_crt_flags(&mut command);
        if !cfg!(windows) {
            command.arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
}

fn build_intel() {
    let source = native_source("intel", "membench-fingerprint-intel.cpp");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_intel_wrapper.cpp",
        &source,
        "mesh_llm_gpu_bench_intel_main",
    );
    let object = out_path("mesh_llm_gpu_bench_intel.o");
    let icpx = std::env::var("ICPX").unwrap_or_else(|_| "icpx".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(icpx);
        command.arg("-O3").arg("-fsycl");
        if !cfg!(windows) {
            command.arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_intel");
    println!("cargo:rustc-link-lib=dylib=sycl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn build_ggml_probe() {
    let workspace_root =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let target = std::env::var("TARGET").unwrap_or_default();
    let build_dir = std::env::var("LLAMA_STAGE_BUILD_DIR")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BUILD_DIR"))
        .map(std::path::PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| default_llama_build_dir(&workspace_root, &target));
    let source_dir = workspace_root.join(".deps/llama.cpp");
    let ggml_header = source_dir.join("ggml/include/ggml.h");
    let ggml_base = build_dir.join("ggml/src/libggml-base.a");
    let ggml = build_dir.join("ggml/src/libggml.a");
    let has_ggml_cpu = any_archive_exists(
        &build_dir,
        &["ggml/src/libggml-cpu.a", "ggml/src/ggml-cpu/libggml-cpu.a"],
    );
    if !ggml_header.exists() || !ggml_base.exists() || !ggml.exists() || !has_ggml_cpu {
        println!(
            "cargo:warning=GGML decode probes disabled; run scripts/prepare-llama.sh and scripts/build-llama.sh first"
        );
        return;
    }

    let has_metal = build_dir
        .join("ggml/src/ggml-metal/libggml-metal.a")
        .exists();
    let has_cuda = build_dir.join("ggml/src/ggml-cuda/libggml-cuda.a").exists();
    let has_hip = build_dir.join("ggml/src/ggml-hip/libggml-hip.a").exists();
    if !has_metal && !has_cuda && !has_hip {
        println!(
            "cargo:warning=GGML decode probes disabled; no accelerated GGML backend archive found"
        );
        return;
    }

    let object = compile_ggml_probe_object(&source_dir, has_metal, has_cuda, has_hip);
    archive_static_lib(&object, "mesh_llm_gpu_bench_ggml_probe");

    println!("cargo:rustc-cfg=mesh_llm_gpu_bench_has_ggml_probe");
    println!("cargo:rerun-if-changed=native/ggml/decode-probe.cpp");
    println!("cargo:rerun-if-changed={}", ggml_header.display());
    for dir in [
        build_dir.join("ggml/src"),
        build_dir.join("ggml/src/ggml-cpu"),
        build_dir.join("ggml/src/ggml-blas"),
        build_dir.join("ggml/src/ggml-cuda"),
        build_dir.join("ggml/src/ggml-hip"),
        build_dir.join("ggml/src/ggml-metal"),
        build_dir.join("ggml/src/ggml-vulkan"),
    ]
    .iter()
    .filter(|dir| dir.exists())
    {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rerun-if-changed={}", object.display());
    link_optional_static(&build_dir, "ggml/src/ggml-cuda/libggml-cuda.a", "ggml-cuda");
    link_optional_static(&build_dir, "ggml/src/ggml-hip/libggml-hip.a", "ggml-hip");
    link_optional_static(
        &build_dir,
        "ggml/src/ggml-metal/libggml-metal.a",
        "ggml-metal",
    );
    link_first_static(
        &build_dir,
        &["ggml/src/libggml-cpu.a", "ggml/src/ggml-cpu/libggml-cpu.a"],
        "ggml-cpu",
    );
    link_first_static(
        &build_dir,
        &[
            "ggml/src/libggml-blas.a",
            "ggml/src/ggml-blas/libggml-blas.a",
        ],
        "ggml-blas",
    );
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if has_metal {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        link_linux_openmp_libs(&build_dir.join("CMakeCache.txt"));
        if has_cuda {
            link_linux_cuda_libs(&build_dir.join("CMakeCache.txt"));
        }
        if has_hip {
            link_linux_hip_libs();
        }
    }
}

fn link_linux_openmp_libs(cmake_cache: &std::path::Path) {
    let Ok(contents) = std::fs::read_to_string(cmake_cache) else {
        return;
    };
    if !cmake_bool_enabled(&contents, "GGML_OPENMP")
        && !cmake_bool_enabled(&contents, "GGML_OPENMP_ENABLED")
    {
        return;
    }

    for lib in openmp_lib_names(&contents) {
        if let Some(path) = cmake_cache_value(&contents, &format!("OpenMP_{lib}_LIBRARY")) {
            let path = std::path::PathBuf::from(path);
            if let Some(parent) = path.parent()
                && parent.is_dir()
            {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
        }
        if lib != "pthread" {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

fn cmake_bool_enabled(contents: &str, key: &str) -> bool {
    cmake_cache_value(contents, key).is_some_and(|value| {
        matches!(
            value.trim().to_ascii_uppercase().as_str(),
            "ON" | "TRUE" | "YES" | "1"
        )
    })
}

fn openmp_lib_names(contents: &str) -> Vec<String> {
    let mut names = Vec::new();
    for key in ["OpenMP_C_LIB_NAMES", "OpenMP_CXX_LIB_NAMES"] {
        if let Some(value) = cmake_cache_value(contents, key) {
            for name in value
                .split(';')
                .map(str::trim)
                .filter(|name| !name.is_empty())
            {
                if !names.iter().any(|existing| existing == name) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

fn compile_ggml_probe_object(
    source_dir: &std::path::Path,
    has_metal: bool,
    has_cuda: bool,
    has_hip: bool,
) -> std::path::PathBuf {
    let object = out_path("mesh_llm_gpu_bench_ggml_probe.o");
    run_or_panic({
        let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
        let mut command = std::process::Command::new(cxx);
        command
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg("-I")
            .arg(source_dir.join("ggml/include"))
            .arg("-I")
            .arg(source_dir.join("ggml/src"))
            .arg("-I")
            .arg(source_dir.join("ggml/src/ggml-cpu"));
        if has_metal {
            command
                .arg("-DMESH_LLM_GGML_PROBE_METAL")
                .arg("-I")
                .arg(source_dir.join("ggml/src/ggml-metal"));
        }
        if has_cuda {
            command
                .arg("-DMESH_LLM_GGML_PROBE_CUDA")
                .arg("-I")
                .arg(source_dir.join("ggml/src/ggml-cuda"));
        }
        if has_hip {
            command
                .arg("-DMESH_LLM_GGML_PROBE_HIP")
                .arg("-I")
                .arg(source_dir.join("ggml/src/ggml-hip"));
        }
        command
            .arg("-c")
            .arg("native/ggml/decode-probe.cpp")
            .arg("-o")
            .arg(&object);
        command
    });
    object
}

fn default_llama_build_dir(workspace_root: &std::path::Path, target: &str) -> std::path::PathBuf {
    let suffix = if target.contains("apple") {
        "metal"
    } else {
        "cpu"
    };
    workspace_root.join(format!(".deps/llama-build/build-stage-abi-{suffix}"))
}

fn link_optional_static(build_dir: &std::path::Path, archive: &str, lib: &str) {
    let path = build_dir.join(archive);
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
        println!("cargo:rustc-link-lib=static={lib}");
    }
}

fn any_archive_exists(build_dir: &std::path::Path, archives: &[&str]) -> bool {
    archives
        .iter()
        .any(|archive| build_dir.join(archive).exists())
}

fn link_first_static(build_dir: &std::path::Path, archives: &[&str], lib: &str) {
    if let Some(path) = archives
        .iter()
        .map(|archive| build_dir.join(archive))
        .find(|path| path.exists())
    {
        println!("cargo:rerun-if-changed={}", path.display());
        println!("cargo:rustc-link-lib=static={lib}");
    }
}

fn link_linux_cuda_libs(cmake_cache: &std::path::Path) {
    for (cache_key, lib) in [
        ("CUDA_cuda_driver_LIBRARY", "cuda"),
        ("CUDA_cudart_LIBRARY", "cudart"),
        ("CUDA_cublas_LIBRARY", "cublas"),
        ("CUDA_cublasLt_LIBRARY", "cublasLt"),
    ] {
        link_linux_lib_from_cache(cmake_cache, cache_key, lib);
    }
    if let Ok(contents) = std::fs::read_to_string(cmake_cache)
        && let Some(nccl_path) = cmake_cache_value(&contents, "NCCL_LIBRARY")
        && !nccl_path.contains("NOTFOUND")
        && !nccl_path.contains("-NOTFOUND")
    {
        let path = std::path::PathBuf::from(&nccl_path);
        if let Some(parent) = path.parent()
            && parent.is_dir()
        {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
        println!("cargo:rustc-link-lib=dylib=nccl");
    }
}

fn link_linux_lib_from_cache(cmake_cache: &std::path::Path, cache_key: &str, lib: &str) {
    if let Ok(contents) = std::fs::read_to_string(cmake_cache)
        && let Some(path) = cmake_cache_value(&contents, cache_key)
    {
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent()
            && parent.is_dir()
        {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

fn cmake_cache_value(contents: &str, key: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let (left, value) = line.split_once('=')?;
        let name = left.split_once(':').map(|(name, _)| name).unwrap_or(left);
        (name == key).then(|| value.to_string())
    })
}

fn link_linux_hip_libs() {
    for root in ["/opt/rocm/lib", "/opt/rocm/lib64"] {
        if std::path::Path::new(root).is_dir() {
            println!("cargo:rustc-link-search=native={root}");
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
