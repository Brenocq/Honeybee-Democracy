CC = nvcc#g++
# Text style
RED    = \033[0;31m
GREEN  = \033[0;32m
NC     = \033[0m
BOLD   = \033[1m

# Folders
BIN	   = bin/
SRC	   = src/
LIB    = lib/
OBJ    = obj/

# Files
FILES = utils main window stbImage scoutBee environment hive nestBox

SOURCES=$(patsubst %, ${SRC}%.cpp, ${FILES})
HEADERS=$(patsubst %, ${SRC}%.h, ${FILES})
OBJECTS=$(patsubst %, ${OBJ}%.o, ${FILES})

#DEPENDENCIES=${LIB}parameters.h

EXECUTABLE=${BIN}beeSimulation

# Flags
FLAGS= -lGL -lGLU -lglfw -lGLEW -I${LIB}
CUDA_FLAGS = -lcurand

# Targets
#${EXECUTABLE}: ${OBJECTS}
#	@/bin/echo -e "${GREEN}${BOLD}----- Creating executable -----${NC}"
#	${CC} ${OBJECTS} -o ${EXECUTABLE} ${FLAGS} 
${EXECUTABLE}: ${OBJECTS}
	@/bin/echo -e "${GREEN}${BOLD}----- Creating executable -----${NC}"
	${CC} -arch=sm_30 ${OBJECTS} -o ${EXECUTABLE} ${FLAGS} ${CUDA_FLAGS}

# Compile project files
#${OBJ}%.o: ${SRC}%.cpp
#	@/bin/echo -e "${GREEN}Compiling $<${NC}"
#	${CC} -c $< -o $@ ${FLAGS} 
${OBJ}%.o: ${SRC}%.cpp
	@/bin/echo -e "${GREEN}Compiling $<${NC}"
	${CC} -x cu -arch=sm_30 -dc $< -o $@ ${FLAGS} ${CUDA_FLAGS}

clean:
	@/bin/echo -e "${GREEN}${BOLD}----- Cleaning project -----${NC}"
	rm -rf ${OBJ}*.o
	rm -rf ${EXECUTABLE}

run: ${EXECUTABLE} ${SOURCES}
	@/bin/echo -e "${GREEN}${BOLD}----- Running ${EXECUTABLE} -----${NC}"
	./${EXECUTABLE}
