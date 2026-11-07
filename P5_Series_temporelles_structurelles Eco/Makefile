ifeq ($(OS),Windows_NT)
LIBRARY = series/libkalman.dll
SHARED_FLAGS = -shared
else
LIBRARY = series/libkalman.so
SHARED_FLAGS = -shared -fPIC
endif

all: $(LIBRARY)

$(LIBRARY): series/kalman.cpp
	g++ -O2 $(SHARED_FLAGS) -o $@ $<

clean:
	rm -f series/libkalman.so series/libkalman.dll
