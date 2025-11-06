{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devshell.url = "github:numtide/devshell";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ self, ... }:

    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];

      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "i686-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        { system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
            config.cudaSupport = true;
            config.cudaVersion = "12";
            overlays = [ inputs.devshell.overlays.default ];
          };

          python =
            let
              file = inputs.self + "/.python-version";
              version = if builtins.pathExists file then builtins.readFile file else "3.13";
              major = builtins.substring 0 1 version;
              minor = builtins.substring 2 2 version;
              packageName = "python${major}${minor}";
            in
            pkgs.${packageName} or pkgs.python313;

          popcorn-cli = pkgs.callPackage ./popcorn-cli.nix { };
        in
        {
          _module.args.pkgs = pkgs;

          devshells.default = {
            packages = [
              pkgs.ruff
              pkgs.pyrefly
              pkgs.llvmPackages_21.clang-unwrapped
              pkgs.taplo
              pkgs.glibc.bin
            ];

            env = [
              {
                name = "CUDA_PATH";
                value = pkgs.cudaPackages.cudatoolkit;
              }
              {
                name = "UV_PYTHON_DOWNLOADS";
                value = "never";
              }
              {
                name = "UV_PYTHON";
                value = python.interpreter;
              }
              {
                name = "PYTHONPATH";
                unset = true;
              }
              {
                name = "UV_NO_SYNC";
                value = "1";
              }
              {
                name = "REPO_ROOT";
                eval = "$(git rev-parse --show-toplevel)";
              }
              {
                name = "EXTRA_LDFLAGS";
                value = "-L/lib -L${pkgs.linuxPackages.nvidiaPackages.stable}/lib -L${pkgs.cudaPackages.cudatoolkit}/lib";
              }
              {
                name = "NVIDIA_PATH";
                value = pkgs.linuxPackages.nvidiaPackages.stable;
              }
              {
                name = "LD_LIBRARY_PATH";
                eval = "$LD_LIBRARY_PATH:${
                  inputs.nixpkgs.lib.makeLibraryPath (
                    with pkgs;
                    [
                      cudaPackages.cudatoolkit.lib
                      linuxPackages.nvidiaPackages.stable
                      stdenv.cc.cc.lib
                      glibc
                      libz
                    ]
                  )
                }:/run/opengl-driver/lib";
              }
              {
                name = "LIBRARY_PATH";
                eval = "${
                  inputs.nixpkgs.lib.makeLibraryPath (
                    with pkgs;
                    [
                      cudaPackages.cudatoolkit.lib
                      linuxPackages.nvidiaPackages.stable
                      stdenv.cc.cc.lib
                      glibc
                    ]
                  )
                }:${pkgs.stdenv.cc.cc}/lib/gcc/x86_64-unknown-linux-gnu/14.3.0";
              }
              {
                name = "TRITON_PTXAS_PATH";
                value = "${pkgs.cudaPackages.cudatoolkit}/bin/ptxas";
              }
              {
                name = "CC";
                value = "${pkgs.stdenv.cc}/bin/gcc";
              }
              {
                name = "CPATH";
                value = "${pkgs.glibc.dev}/include";
              }
              {
                name = "PATH";
                eval = "/home/ethanthoma/.local/bin:${pkgs.cudaPackages.cudatoolkit}/bin:$PATH";
              }
            ];

            commands = [
              { package = pkgs.uv; }
              {
                name = "claude";
                package = pkgs.claude-code;
              }
              { package = pkgs.tokei; }
              { package = pkgs.codex; }
              { package = popcorn-cli; }
            ];
          };
        };
    };

  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
}
