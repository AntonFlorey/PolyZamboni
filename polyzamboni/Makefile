all: linux macos windows

linux: linux_x86_64

linux_x86_64:
	pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=3.13 --platform=manylinux_2_17_x86_64

macos: macos_arm64 macos_x86_64

macos_arm64:
	pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=3.13 --platform=macosx_11_0_arm64

macos_x86_64:
	pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=3.13 --platform=macosx_11_0_x86_64

windows: windows_amd64

windows_amd64:
	pip download -r requirements.txt --dest ./wheels --only-binary=:all: --python-version=3.13 --platform=win_amd64
