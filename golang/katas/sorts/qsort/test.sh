#!/bin/sh

go test .
go test -bench=. -benchtime=30x