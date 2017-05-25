TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    energy.cpp \
    basis.cpp
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp
LIBS += -larmadillo -llapack -lblas
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3

HEADERS += \
    basis.h \
    maths.h
