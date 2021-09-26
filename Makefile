sudo = /usr/bin/sudo
pip_inst = $(sudo) /usr/bin/pip3 install
rm = /bin/rm

py_pkgs = emcee corner dill numdifftools lmfit matplotlib tables \
numexpr statsmodels scipy numpy pandas patsy pyparsing pillow

install:
	$(pip_inst) $(py_pkgs)

clean:
	$(rm) *.png *.svg

.PHONY: clean
