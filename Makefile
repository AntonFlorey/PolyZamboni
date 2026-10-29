# Blender 5.1 bundles Python 3.13, Blender 5.2 bundles Python 3.14.
# Binary wheels must be downloaded for both versions.
PY_VERSIONS = 3.13 3.14

all: linux macos windows

linux: linux_x86_64

linux_x86_64:
	for v in $(PY_VERSIONS); do \
		pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=$$v --platform=manylinux_2_17_x86_64 --platform=manylinux_2_28_x86_64; \
	done

macos: macos_arm64 macos_x86_64

macos_arm64:
	for v in $(PY_VERSIONS); do \
		pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=$$v --platform=macosx_11_0_arm64; \
	done

macos_x86_64:
	for v in $(PY_VERSIONS); do \
		pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=$$v --platform=macosx_10_13_x86_64; \
	done

windows: windows_amd64

windows_amd64:
	for v in $(PY_VERSIONS); do \
		pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=$$v --platform=win_amd64; \
	done
