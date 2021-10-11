CC=g++
CFLAGS=-c -I$(INCLUDE_DIR) --std=c++14
CFLAGS_OPTIMISATIONS=-O2 -DNDEBUG -ffast-math
SRC_DIR=src
BUILD_DIR=build
INCLUDE_DIR=include

SOURCES=$(wildcard $(SRC_DIR)/*.cpp)
OBJECTS=$(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
EXECUTABLE=exe

.PHONY: all
all: release

$(BUILD_DIR)/$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: profile
profile: CFLAGS += -g $(CFLAGS_OPTIMISATIONS)
profile: LDFLAGS += -g -lm
profile: $(BUILD_DIR) $(BUILD_DIR)/$(EXECUTABLE)

.PHONY: release
release: CFLAGS += $(CFLAGS_OPTIMISATIONS)
release: $(BUILD_DIR) $(BUILD_DIR)/$(EXECUTABLE)

.PHONY: debug
debug: CFLAGS += -DDEBUG -g -pg -Wall
debug: LDFLAGS += -pg -lm
debug: $(BUILD_DIR) $(BUILD_DIR)/$(EXECUTABLE)
