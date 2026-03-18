import { useState, useEffect, useMemo } from 'react'
import { Play, Square, Trash2, Download, Monitor, Clock, CheckCircle, XCircle, Loader2, Info } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'

interface TrainingPageProps {
  wsMessages: any[]
}

const defaultConfig = {
  dataset_id: '',
  model_id: 'Qwen/Qwen3-0.6B',
  method: 'lora',
  data_format: 'chat',
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  epochs: 3,
  batch_size: 4,
  gradient_accumulation_steps: 4,
  learning_rate: 2e-4,
  lr_scheduler: 'cosine',
  warmup_ratio: 0.05,
  weight_decay: 0.01,
  max_seq_length: 2048,
  val_split: 0.1,
  save_steps: 100,
  eval_steps: 50,
  early_stopping_patience: 5,
  fp16: true,
  bf16: false,
  gradient_checkpointing: true,
  quant_bits: 4,
  output_name: '',
  merge_adapter: false,
}

const tooltips: Record<string, string> = {
  lora_r: 'LoRA rank — higher = more parameters, more expressive but slower. 8-64 typical.',
  lora_alpha: 'LoRA scaling factor. Usually 2x the rank.',
  lora_dropout: 'Dropout for LoRA layers. Helps prevent overfitting.',
  epochs: 'Number of full passes through the training data.',
  batch_size: 'Samples per GPU per step. Lower if OOM.',
  gradient_accumulation_steps: 'Accumulate gradients over N steps. Effective batch = batch_size × this.',
  learning_rate: 'Step size for optimization. 1e-4 to 3e-4 typical for LoRA.',
  lr_scheduler: 'How learning rate changes during training.',
  warmup_ratio: 'Fraction of training spent warming up the learning rate.',
  max_seq_length: 'Maximum token sequence length. Longer = more VRAM.',
  val_split: 'Fraction of data held out for validation.',
  early_stopping_patience: 'Stop after N evals without improvement. 0 = disabled.',
  quant_bits: 'Quantization bits for QLoRA. 4-bit is most VRAM efficient.',
}

export default function TrainingPage({ wsMessages }: TrainingPageProps) {
  const [config, setConfig] = useState(defaultConfig)
  const [models, setModels] = useState<any[]>([])
  const [datasets, setDatasets] = useState<any[]>([])
  const [jobs, setJobs] = useState<any[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [gpu, setGpu] = useState<any>(null)
  const [compareIds, setCompareIds] = useState<string[]>([])

  useEffect(() => {
    api.getModels().then(r => setModels(r.models))
    api.getDatasets().then(r => setDatasets(r.datasets))
    loadJobs()
    api.getGpuStatus().then(setGpu).catch(() => {})
  }, [])

  // Process WebSocket messages for real-time metrics
  useEffect(() => {
    const last = wsMessages[wsMessages.length - 1]
    if (!last) return
    if (last.type === 'job_update' || last.type === 'metrics') {
      loadJobs()
    }
  }, [wsMessages])

  const loadJobs = async () => {
    const res = await api.getJobs()
    setJobs(res.jobs)
  }

  const startTraining = async () => {
    if (!config.dataset_id) return alert('Select a dataset first')
    const res = await api.startTraining(config)
    setSelectedJobId(res.job_id)
    loadJobs()
  }

  const stopJob = async (id: string) => {
    await api.stopJob(id)
    loadJobs()
  }

  const deleteJob = async (id: string) => {
    if (!confirm('Delete this job and its outputs?')) return
    await api.deleteJob(id)
    if (selectedJobId === id) setSelectedJobId(null)
    loadJobs()
  }

  const exportModel = async (id: string, merge: boolean) => {
    await api.exportModel(id, merge)
    alert('Model exported successfully!')
  }

  const selectedJob = jobs.find(j => j.id === selectedJobId)

  const statusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
      case 'completed': return <CheckCircle className="w-4 h-4 text-emerald-400" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />
      case 'stopped': return <Square className="w-4 h-4 text-yellow-400" />
      default: return <Clock className="w-4 h-4 text-muted-foreground" />
    }
  }

  const comparisonData = useMemo(() => {
    if (compareIds.length < 2) return null
    const jobsToCompare = compareIds.map(id => jobs.find(j => j.id === id)).filter(Boolean)
    if (jobsToCompare.length < 2) return null
    const maxSteps = Math.max(...jobsToCompare.map((j: any) => j.metrics?.length || 0))
    const data = []
    for (let i = 0; i < maxSteps; i++) {
      const point: any = { step: i }
      jobsToCompare.forEach((j: any, idx: number) => {
        if (j.metrics?.[i]) {
          point[`loss_${j.name}`] = j.metrics[i].loss
        }
      })
      data.push(point)
    }
    return { data, jobs: jobsToCompare }
  }, [compareIds, jobs])

  const updateConfig = (key: string, value: any) => setConfig(prev => ({ ...prev, [key]: value }))

  const ConfigField = ({ label, field, type = 'number', options, ...props }: any) => (
    <div className="space-y-1">
      <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
        {label}
        {tooltips[field] && (
          <span title={tooltips[field]} className="cursor-help"><Info className="w-3 h-3" /></span>
        )}
      </label>
      {options ? (
        <select
          className="w-full rounded-md border bg-background px-3 py-2 text-sm"
          value={(config as any)[field]}
          onChange={e => updateConfig(field, e.target.value)}
        >
          {options.map((o: any) => <option key={o.value || o} value={o.value || o}>{o.label || o}</option>)}
        </select>
      ) : type === 'checkbox' ? (
        <input
          type="checkbox"
          checked={(config as any)[field]}
          onChange={e => updateConfig(field, e.target.checked)}
          className="rounded"
        />
      ) : (
        <Input
          type={type}
          value={(config as any)[field]}
          onChange={e => updateConfig(field, type === 'number' ? Number(e.target.value) : e.target.value)}
          step={type === 'number' && field.includes('rate') || field.includes('ratio') || field.includes('dropout') ? 0.001 : undefined}
          {...props}
        />
      )}
    </div>
  )

  return (
    <div className="flex h-full">
      {/* Config Panel */}
      <div className="w-96 border-r overflow-auto p-4 space-y-4">
        <h2 className="text-lg font-semibold">Training Configuration</h2>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Model & Data</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <ConfigField label="Model" field="model_id" options={models.map(m => ({ value: m.id, label: `${m.name} (${m.params})` }))} />
            <ConfigField label="Dataset" field="dataset_id" options={[
              { value: '', label: 'Select dataset...' },
              ...datasets.map(d => ({ value: d.id, label: `${d.name} (${d.example_count})` }))
            ]} />
            <ConfigField label="Method" field="method" options={[
              { value: 'lora', label: 'LoRA' },
              { value: 'qlora', label: 'QLoRA (quantized)' },
            ]} />
            <ConfigField label="Data Format" field="data_format" options={[
              { value: 'chat', label: 'Chat / Instruction' },
              { value: 'raw', label: 'Raw Text' },
            ]} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">LoRA Parameters</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            <ConfigField label="Rank (r)" field="lora_r" />
            <ConfigField label="Alpha" field="lora_alpha" />
            <ConfigField label="Dropout" field="lora_dropout" />
            {config.method === 'qlora' && <ConfigField label="Quant Bits" field="quant_bits" options={[{ value: 4, label: '4-bit' }, { value: 8, label: '8-bit' }]} />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Training Parameters</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            <ConfigField label="Epochs" field="epochs" />
            <ConfigField label="Batch Size" field="batch_size" />
            <ConfigField label="Grad Accum Steps" field="gradient_accumulation_steps" />
            <ConfigField label="Learning Rate" field="learning_rate" />
            <ConfigField label="LR Scheduler" field="lr_scheduler" options={['cosine', 'linear', 'constant', 'constant_with_warmup']} />
            <ConfigField label="Warmup Ratio" field="warmup_ratio" />
            <ConfigField label="Weight Decay" field="weight_decay" />
            <ConfigField label="Max Seq Length" field="max_seq_length" />
            <ConfigField label="Val Split" field="val_split" />
            <ConfigField label="Save Steps" field="save_steps" />
            <ConfigField label="Eval Steps" field="eval_steps" />
            <ConfigField label="Early Stop Patience" field="early_stopping_patience" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Options</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.fp16} onChange={e => updateConfig('fp16', e.target.checked)} className="rounded" /> FP16
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.bf16} onChange={e => { updateConfig('bf16', e.target.checked); if (e.target.checked) updateConfig('fp16', false) }} className="rounded" /> BF16
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.gradient_checkpointing} onChange={e => updateConfig('gradient_checkpointing', e.target.checked)} className="rounded" /> Grad Ckpt
              </label>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={config.merge_adapter} onChange={e => updateConfig('merge_adapter', e.target.checked)} className="rounded" />
              Merge adapter after training
            </label>
            <ConfigField label="Run Name (optional)" field="output_name" type="text" />
          </CardContent>
        </Card>

        <Button className="w-full" size="lg" onClick={startTraining} disabled={!config.dataset_id}>
          <Play className="w-4 h-4 mr-2" /> Start Training
        </Button>
      </div>

      {/* Jobs & Metrics */}
      <div className="flex-1 flex flex-col overflow-auto">
        {/* GPU Status */}
        {gpu && (
          <div className="border-b p-4">
            <div className="flex items-center gap-4 flex-wrap">
              <Monitor className="w-4 h-4 text-muted-foreground" />
              <Badge variant={gpu.device_type === 'cpu' ? 'secondary' : 'success'}>
                {gpu.device_type?.toUpperCase() || 'Unknown'}
              </Badge>
              {gpu.gpus?.length > 0 ? gpu.gpus.map((g: any) => (
                <div key={g.id} className="flex items-center gap-3 text-sm">
                  <span className="font-medium">{g.name}</span>
                  {g.memory_total != null && (
                    <>
                      <span>{g.memory_used ?? '?'}MB / {g.memory_total}MB</span>
                      <div className="w-24 h-2 bg-secondary rounded-full overflow-hidden">
                        <div className="h-full bg-primary rounded-full" style={{ width: `${((g.memory_used ?? 0) / g.memory_total) * 100}%` }} />
                      </div>
                    </>
                  )}
                  {g.gpu_util != null && <span>{g.gpu_util.toFixed(0)}% util</span>}
                </div>
              )) : gpu.device_type === 'cpu' ? (
                <span className="text-sm text-muted-foreground">No GPU detected — running on CPU</span>
              ) : null}
            </div>
          </div>
        )}

        {/* Job List */}
        <div className="border-b p-4">
          <h3 className="font-semibold mb-3">Jobs</h3>
          <div className="space-y-2">
            {jobs.length === 0 && <p className="text-sm text-muted-foreground">No training jobs yet. Configure and start one!</p>}
            {jobs.map(job => (
              <div
                key={job.id}
                className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                  selectedJobId === job.id ? 'bg-primary/10 border border-primary/20' : 'bg-card hover:bg-accent'
                }`}
                onClick={() => setSelectedJobId(job.id)}
              >
                {statusIcon(job.status)}
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">{job.name}</div>
                  <div className="text-xs text-muted-foreground">{job.config?.model_id?.split('/')[1]} · {job.config?.method}</div>
                </div>
                <Badge variant={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'destructive' : 'secondary'}>
                  {job.status}
                </Badge>
                <div className="flex gap-1">
                  {job.status === 'running' && (
                    <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); stopJob(job.id) }}>
                      <Square className="w-3.5 h-3.5" />
                    </Button>
                  )}
                  {job.status === 'completed' && (
                    <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); exportModel(job.id, true) }}>
                      <Download className="w-3.5 h-3.5" />
                    </Button>
                  )}
                  <Button size="sm" variant="ghost" className="text-destructive" onClick={e => { e.stopPropagation(); deleteJob(job.id) }}>
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                  <input
                    type="checkbox"
                    checked={compareIds.includes(job.id)}
                    onChange={e => {
                      e.stopPropagation()
                      setCompareIds(prev => prev.includes(job.id) ? prev.filter(id => id !== job.id) : [...prev, job.id])
                    }}
                    title="Compare"
                    className="rounded ml-2"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics Charts */}
        <div className="flex-1 p-4 space-y-4">
          {selectedJob && selectedJob.metrics?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Training Metrics — {selectedJob.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={selectedJob.metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="Train Loss" />
                    <Line type="monotone" dataKey="eval_loss" stroke="#10b981" strokeWidth={2} dot={false} name="Val Loss" />
                    <Line type="monotone" dataKey="learning_rate" stroke="#f59e0b" strokeWidth={1} dot={false} name="LR" yAxisId={0} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Comparison Chart */}
          {comparisonData && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Run Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={comparisonData.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
                    <Legend />
                    {comparisonData.jobs.map((j: any, i: number) => (
                      <Line key={j.id} type="monotone" dataKey={`loss_${j.name}`} stroke={['hsl(var(--primary))', '#10b981', '#f59e0b', '#ef4444'][i % 4]} strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {selectedJob?.error && (
            <Card className="border-destructive">
              <CardHeader>
                <CardTitle className="text-sm text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-xs text-red-400 whitespace-pre-wrap overflow-auto max-h-48">{selectedJob.error}</pre>
              </CardContent>
            </Card>
          )}

          {!selectedJob && jobs.length > 0 && (
            <div className="flex items-center justify-center h-48 text-muted-foreground">
              Select a job to view metrics
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
