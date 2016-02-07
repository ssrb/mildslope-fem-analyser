# Copyright (c) 2015, Sebastien Sydney Robert Bigot
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
V = 0

AT_0 := @
AT_1 :=
AT = $(AT_$(V))

REDIR_0 := > /dev/null
REDIR_1 :=
REDIR = $(REDIR_$(V))

CXX := g++
CXXFLAGS := -Wall -Wextra -m64 -O3 $(shell mpic++ --showme:compile) \
	-I/usr/local/cuda/include
# -mkl=parallel

LDFLAGS :=  $(shell mpic++ --showme:link) -lblas \
	-L/usr/local/cuda/lib64 \
	-L/usr/local/cuda/lib \
	-ldl -lnsl -lutil \
	-lcudart \
	-lcusparse 

# -pthread -openmp -mkl=parallel \
# -llmpe -lmpe
# -lopen-rte -lopen-pal 

NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O3 -arch=sm_20 -Xcompiler -m64 -Xcompiler -Wall -Xcompiler -Wextra


CPPSRC= $(wildcard *.cpp) $(shell find tinyxml -name "*.cpp")
CUDASRC=$(shell find . -name "*.cu")

OUT=obj
OBJS=$(CPPSRC:%.cpp=$(OUT)/%.o) $(CUDASRC:%.cu=$(OUT)/%.o) 
DEPS=$(CPPSRC:%.cpp=$(OUT)/%.d)

TARGET=babyharesED

NODEPS=clean

.PHONY = all clean

.SECONDEXPANSION:

all: $(TARGET)

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
    -include $(DEPS)
endif

$(TARGET): $(OBJS)
	@echo [LD] $@
	$(AT)$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@ $(REDIR)

$(OUT)/%.d:%.cpp $$(@D)/.f
	@echo [DEP] $<
	$(AT)$(CXX) $(CXXFLAGS) -MM -MT '$(patsubst %.d,%.o,$@)' $< -MF $@ $(REDIR)

$(OUT)/%.o:%.cpp $(OUT)/%.d
	@echo [C++] $<
	$(AT)$(CXX) $< $(CXXFLAGS) -c -o $@ $(REDIR)

$(OUT)/%.o:%.cu
	@echo [CUDA] $<
	$(AT)$(NVCC) $(NVCCFLAGS) $< -c -o $@ $(REDIR)

%/.f:
	@echo [MKDIR] $(dir $@)
	$(AT)mkdir -p $(dir $@) 
	$(AT)touch $@

.PRECIOUS: %/.f

clean:
	@echo [CLEAN]
	$(AT)-rm -rf $(TARGET) $(OUT)
