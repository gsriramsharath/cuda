
TARGET=inst_pipeline

all: $(TARGET)

$(TARGET):
	$(MAKE) -C src
	mkdir -p bin; cp src/$(TARGET) bin

clean:
	$(MAKE) clean -C src
	rm -f bin/$(TARGET)
