sudo = /usr/bin/sudo
pip_inst = $(sudo) /usr/bin/pip3 install
rm = /bin/rm -fr

py_pkgs = emcee corner dill numdifftools lmfit matplotlib tables \
numexpr statsmodels scipy numpy pandas patsy pyparsing pillow

test: clean
	./nlin_fit.py
	./nlin_fit.py -t

install:
	$(pip_inst) $(py_pkgs)

.PHONY: clean

clean:
	$(rm) *.png *.svg


