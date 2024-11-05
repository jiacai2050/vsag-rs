
.PHONY: fmt
fmt:
	find src include -iname "*.h" -o -iname "*.cpp" | xargs clang-format -i

.PHONY: test
test:
	cargo test
