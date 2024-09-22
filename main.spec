block_cipher = None

pyfiles = ['main.py',
    'track/utils/__init__.py', 'track/utils/gmc.py', 'track/utils/kalman_filter.py', 'track/utils/matching.py',
    'track/__init__.py', 'track/basetrack.py', 'track/bytetracker.py', 'track/botsort.py', 'track/track.py',
    'utils/__init__.py', 'utils/load.py', 'utils/process.py', 'utils/results.py']


a = Analysis(pyfiles,
             pathex=['D:\\Code\\python\\torchless-yolotrack'],
             excludes=['demo.py'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='torchless-track',		# 打包程序的名字
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )