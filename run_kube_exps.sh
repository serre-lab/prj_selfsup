#!/usr/bin/env bash


kubectl create -f kube_files/ar_ar_32.yaml
kubectl create -f kube_files/ar_False_32.yaml
kubectl create -f kube_files/False_ar_32.yaml
kubectl create -f kube_files/False_False_32.yaml
