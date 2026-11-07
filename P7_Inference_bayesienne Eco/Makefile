all: cpp rust

cpp: bayes/libbayes_cpp.so

bayes/libbayes_cpp.so: cpp/bayes.cpp
	g++ -O2 -ffp-contract=off -shared -fPIC -Wall -o $@ $<

rust: bayes/libbayes_rust.so

bayes/libbayes_rust.so: rust/src/lib.rs rust/Cargo.toml
	cargo build --release --manifest-path rust/Cargo.toml
	cp rust/target/release/libbayes_rust.so $@

clean:
	rm -f bayes/*.so
	rm -rf rust/target rust/Cargo.lock
