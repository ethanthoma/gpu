{
  lib,
  rustPlatform,
  fetchFromGitHub,
  pkg-config,
  openssl,
  nix-update-script,
}:

let
  version = "1.1.44";
in
rustPlatform.buildRustPackage {
  pname = "popcorn-cli";
  inherit version;

  src = fetchFromGitHub {
    owner = "gpu-mode";
    repo = "popcorn-cli";
    rev = "v${version}";
    hash = "sha256-iIwU/FSveNLO5ZD0g8GfKSGLUstZA/akBg/1F1BOIQY=";
  };

  cargoHash = "sha256-eMhhoONOUNRDx+vxzkcv9AE2XE3mQ8XLH2QqlgDbXeI=";

  nativeBuildInputs = [ pkg-config ];
  buildInputs = [ openssl ];

  # Version check disabled: binary reports 0.1.0 while git tag is v1.1.44
  # nativeInstallCheckInputs = [ versionCheckHook ];
  # versionCheckProgramArg = "--version";
  # doInstallCheck = true;

  passthru.updateScript = nix-update-script { };

  meta = {
    description = "CLI for submitting solutions to the Popcorn Discord Bot";
    homepage = "https://github.com/gpu-mode/popcorn-cli";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ ];
    mainProgram = "popcorn-cli";
  };
}
