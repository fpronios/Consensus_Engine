# -*- mode: python -*-

block_cipher = None


a = Analysis(['helper_functions.py'],
             pathex=['C:\\Users\\Filippos\\PycharmProjects\\Consensus_Engine'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy.scipy',
                            'scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.integrate._dop',
                            'scipy.integrate.lsoda'
                            'sklearn.metrics',
                            'sklearn.sklearn',
                            'sklearn.metrics._roc_curve',
                            'sklearn.metrics._auc',
                            'sklearn.preprocessing._label_binarize',
                            'sklearn',],
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
          name='helper_functions',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='app_icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='helper_functions')
