# -*- mode: python -*-

block_cipher = None


a = Analysis(['brutereg.py'],
             pathex=['./modules/', './lib/python2.7/site-packages/', '/dls/science/groups/i04-1/software/BruteReg/BruteReg'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='brutereg',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='brutereg')
