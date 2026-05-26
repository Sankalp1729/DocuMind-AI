{{- define "documind.name" -}}
documind
{{- end -}}

{{- define "documind.namespace" -}}
{{ .Values.global.namespace | default "documind" }}
{{- end -}}

{{- define "documind.labels" -}}
app.kubernetes.io/name: {{ include "documind.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
