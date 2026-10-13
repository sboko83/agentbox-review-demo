import { useCallback, useMemo, useState } from 'react'

import type { AgentSpec, MemoryLayer } from './api/sessions'
import { AgentPanel, type PanelTab } from './components/AgentPanel'
import { ChatPane } from './components/ChatPane'
import { ConfigurationEditor } from './components/ConfigurationEditor'
import { Experiment } from './components/Experiment'
import { InvariantEditor } from './components/InvariantEditor'
import { PresetLibrary } from './components/PresetLibrary'
import { ProfilePanel } from './components/ProfilePanel'
import { SavedExperiments } from './components/SavedExperiments'
import { prepareExperiment, type SavedExperiment } from './components/experimentRecipe'
import { Sidebar, type Screen } from './components/Sidebar'
import { schemaDefaults, type SpecValues } from './components/agentSpec'
import { useChatSession } from './hooks/useChatSession'
import { usePresets } from './hooks/usePresets'
import { useUserProfiles } from './hooks/useUserProfiles'
import { useSettings } from './hooks/useSettings'

const SCREEN_TITLE: Record<Screen, string> = { chat: 'Чат', configuration: 'Конфигурация', library: 'Библиотека', experiment: 'Матричный прогон' }

export default function App() {
  const chat = useChatSession()
  const presetStore = usePresets()
  const profileStore = useUserProfiles()
  const { schema, models, error: settingsError } = useSettings()
  const [screen, setScreen] = useState<Screen>('chat')
  const [libraryTab, setLibraryTab] = useState<'presets' | 'profiles' | 'experiments'>('presets')
  const [panel, setPanel] = useState<PanelTab | null>(null)
  const [lastPanel, setLastPanel] = useState<PanelTab>(() => {
    try {
      const saved = localStorage.getItem('agentbox.inspector-tab')
      return saved === 'memory' || saved === 'profile' ? saved : 'context'
    } catch { return 'context' }
  })
  const [rail, setRail] = useState(() => window.innerWidth > 900)
  const [newProfileId, setNewProfileId] = useState<string | null | undefined>(undefined)
  const [configDirty, setConfigDirty] = useState(false)
  const [invariantsDirty, setInvariantsDirty] = useState(false)
  const [presetDirty, setPresetDirty] = useState(false)
  const [profileDirty, setProfileDirty] = useState(false)
  const [configEpoch, setConfigEpoch] = useState(0)
  const [updating, setUpdating] = useState(false)
  const [error, setError] = useState('')
  const busy = chat.busy || chat.creating || updating
  const editable = chat.session?.editable ?? true
  const current = (chat.session?.spec ?? chat.draftSpec ?? presetStore.presets.find((item) => item.is_default)?.spec
    ?? (schema ? schemaDefaults(schema) : null)) as AgentSpec | null
  const profileId = chat.session ? chat.session.user_profile_id
    : newProfileId === undefined ? profileStore.profiles.find((item) => item.is_default)?.id ?? null : newProfileId
  const profile = profileStore.profiles.find((item) => item.id === profileId)?.profile ?? null
  const presetId = presetStore.presets.find((item) => JSON.stringify(item.spec) === JSON.stringify(current))?.id ?? ''

  const { decideMemory, finishWorkingTask, forgetItem, moveItem, rememberItem, updateMemory } = chat
  const { updateSpec } = chat
  const memoryActions = useMemo(() => ({
    onRemember: rememberItem, onForget: forgetItem, onMove: moveItem,
    onClear: (layer: MemoryLayer) => updateMemory(layer, {}), onFinish: finishWorkingTask, onDecide: decideMemory,
  }), [decideMemory, finishWorkingTask, forgetItem, moveItem, rememberItem, updateMemory])

  const changedProfile = async (deletedProfileId?: string) => {
    if (deletedProfileId && deletedProfileId === newProfileId) {
      setNewProfileId(null)
      chat.setDraftUserProfile(null)
    }
    await chat.reloadSession()
    await chat.refreshSessions()
  }
  function navigate(next: Screen) {
    setScreen(next); setPanel(null)
    if (window.innerWidth <= 900) setRail(false)
  }
  function openPanel(tab: PanelTab) {
    setLastPanel(tab); setPanel(tab)
    try { localStorage.setItem('agentbox.inspector-tab', tab) } catch { /* Preference only. */ }
  }
  function discardConfiguration() {
    if ((configDirty || invariantsDirty) && !window.confirm('Отменить неприменённые изменения конфигурации?')) return false
    setConfigDirty(false); setInvariantsDirty(false); setConfigEpoch((value) => value + 1)
    return true
  }
  async function act(action: () => Promise<void>) {
    if (busy) return
    setUpdating(true); setError('')
    try { await action() } catch (cause) { setError(String(cause)) }
    finally { setUpdating(false) }
  }
  const applySpec = useCallback(async (spec: SpecValues) => {
    setUpdating(true)
    try { await updateSpec(spec) } finally { setUpdating(false) }
  }, [updateSpec])
  async function selectProfile(id: string | null) {
    if (chat.session) await chat.changeUserProfile(id)
    else { setNewProfileId(id); chat.setDraftUserProfile(id) }
  }
  async function startExperiment(recipe: SavedExperiment) {
    if (busy || !discardConfiguration()) return
    setUpdating(true)
    try {
      const prepared = await prepareExperiment(recipe)
      chat.loadExperiment({ ...prepared.spec }, prepared.profileId, prepared.session)
      setNewProfileId(prepared.profileId)
      await profileStore.refresh()
      navigate('chat')
    } finally { setUpdating(false) }
  }
  const inspectorVisible = screen === 'chat' && panel !== null
  const shell = ['shell', rail ? '' : 'shell-no-rail', inspectorVisible ? '' : 'shell-no-panel'].filter(Boolean).join(' ')

  return (
    <div className={shell}>
      {rail && <Sidebar sessions={chat.sessions} currentId={chat.session?.id ?? null} screen={screen} busy={busy}
        onOpen={(id) => { if (discardConfiguration()) { navigate('chat'); void chat.open(id) } }}
        onDrop={(id) => { if (id !== chat.session?.id || discardConfiguration()) void chat.drop(id) }}
        onRefresh={() => void chat.refreshSessions()}
        onNew={() => { if (discardConfiguration()) { navigate('chat'); void chat.newDraft() } }}
        onExperiment={() => navigate('experiment')} onConfiguration={() => navigate('configuration')} onLibrary={() => navigate('library')} />}
      {rail && <button type="button" className="navigation-scrim" aria-label="Закрыть навигацию" onClick={() => setRail(false)} />}
      <main className="workspace">
        <header className="topbar">
          <button type="button" className="link" onClick={() => setRail((open) => !open)} aria-label={rail ? 'скрыть сессии' : 'показать сессии'}>☰</button>
          {screen !== 'chat' && <button type="button" className="link" onClick={() => navigate('chat')}>← Чат</button>}
          <span className="screen-title">{SCREEN_TITLE[screen]}</span>
          <div className="topbar-right">
            {screen === 'chat' && <>
              <button type="button" className="spend" onClick={() => openPanel('context')} aria-label="Расход сессии">
                <span className="muted">потрачено</span><strong>${(chat.session?.totals.cost_usd ?? 0).toFixed(6)}</strong>
              </button>
              <button type="button" className="link" aria-expanded={inspectorVisible} onClick={() => panel ? setPanel(null) : openPanel(lastPanel)}>Диагностика</button>
            </>}
          </div>
        </header>

        <div className="chat-screen" hidden={screen !== 'chat'}>
          {current && <div className="quick-settings" aria-label="Настройки текущей сессии">
            <label>Пресет<select aria-label="Пресет сессии" value={presetId} disabled={busy || !editable}
              onChange={(event) => { const preset = presetStore.presets.find((item) => item.id === event.target.value); if (preset && discardConfiguration()) void act(() => chat.updateSpec({ ...preset.spec })) }}>
              <option value="">Своя конфигурация</option>{presetStore.presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
            </select></label>
            <label>Модель<select aria-label="Модель сессии" value={`${current.model.provider}/${current.model.model}`} disabled={busy || !editable}
              onChange={(event) => { const selected = models.find((model) => `${model.provider}/${model.id}` === event.target.value); if (selected && discardConfiguration()) void act(() => chat.updateSpec({ ...current, model: { ...current.model, provider: selected.provider, model: selected.id } })) }}>
              {!models.some((model) => model.provider === current.model.provider && model.id === current.model.model) && <option value={`${current.model.provider}/${current.model.model}`}>{current.model.model}</option>}
              {models.filter((model) => model.offered || (model.provider === current.model.provider && model.id === current.model.model))
                .map((model) => <option key={`${model.provider}/${model.id}`} value={`${model.provider}/${model.id}`}>{model.provider} / {model.id}</option>)}
            </select></label>
            <label>Профиль<select aria-label="Профиль пользователя сессии" value={profileId ?? ''} disabled={busy || !editable} onChange={(event) => void act(() => selectProfile(event.target.value || null))}>
              <option value="">Без профиля</option>{profileStore.profiles.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select></label>
            <button type="button" onClick={() => navigate('configuration')}>Настроить{configDirty || invariantsDirty ? ' •' : ''}</button>
            <button type="button" className="memory-status" onClick={() => openPanel('memory')}>
              Память: {current.memory.working_enabled ? 'рабочая' : ''}{current.memory.working_enabled && current.memory.longterm_enabled ? ' + ' : ''}{current.memory.longterm_enabled ? 'долговременная' : ''}{!current.memory.working_enabled && !current.memory.longterm_enabled ? 'выключена' : ''}
            </button>
          </div>}
          {(configDirty || invariantsDirty) && <p className="draft-notice">Есть неприменённые настройки. Чат использует сохранённую конфигурацию. <button type="button" className="link" onClick={() => navigate('configuration')}>Открыть изменения</button></p>}
          <ChatPane chat={{ ...chat, busy }} modelWindow={models.find((model) => model.provider === current?.model.provider && model.id === current?.model.model)?.context_window ?? null} onOpenContext={() => openPanel('context')} />
        </div>

        <div className="screen-scroll settings-screen" hidden={screen !== 'configuration'}>
          {schema && current ? <ConfigurationEditor key={`configuration-${chat.session?.id ?? `draft-${chat.draftVersion}`}-${configEpoch}`} schema={schema} models={models} value={{ ...current }}
            scope={chat.session ? `Текущая сессия: ${chat.session.title || chat.session.id.slice(0, 8)}` : 'Новая сессия'} disabled={busy || !editable}
            onApply={applySpec} onDirty={setConfigDirty} /> : <p>{settingsError || 'Загружаю настройки…'}</p>}
          <InvariantEditor key={`invariants-${chat.session?.id ?? 'draft'}-${configEpoch}`}
            session={chat.session} busy={busy || !editable} onApply={chat.updateInvariants} onDirty={setInvariantsDirty} />
        </div>

        <div className="screen-scroll settings-screen" hidden={screen !== 'library'}>
          <nav className="library-tabs" aria-label="Разделы библиотеки">
            {(['presets', 'profiles', 'experiments'] as const).map((tab) => <button type="button" key={tab} aria-current={libraryTab === tab ? 'page' : undefined} onClick={() => setLibraryTab(tab)}>
              {{ presets: 'Пресеты', profiles: 'Профили', experiments: 'Эксперименты' }[tab]}{(tab === 'presets' && presetDirty || tab === 'profiles' && profileDirty) ? ' •' : ''}
            </button>)}
          </nav>
          <div hidden={libraryTab !== 'presets'}>{schema && current && <PresetLibrary store={presetStore} schema={schema} models={models} current={{ ...current }} onDirty={setPresetDirty} />}</div>
          <div hidden={libraryTab !== 'profiles'}><ProfilePanel session={chat.session} sessions={chat.sessions} store={profileStore} busy={busy} onSessionProfile={selectProfile} onProfileChanged={changedProfile} onDecide={memoryActions.onDecide} onDirty={setProfileDirty} /></div>
          <div hidden={libraryTab !== 'experiments'}>{current && <SavedExperiments key={chat.session?.id ?? 'draft'} spec={current} profile={profile} session={chat.session} busy={busy} onStart={startExperiment} />}</div>
        </div>

        <div className="screen-scroll" hidden={screen !== 'experiment'}>
          <Experiment presets={presetStore.presets} profiles={profileStore.profiles} invariants={chat.session?.invariants.rules ?? []} />
        </div>
        {error && <p className="error app-error" role="alert">{error}<button type="button" className="link" onClick={() => setError('')}>Закрыть</button></p>}
      </main>
      {inspectorVisible && <>
        <button type="button" className="inspector-scrim" aria-label="Закрыть диагностику" onClick={() => setPanel(null)} />
        <AgentPanel session={chat.session} models={models} busy={busy} tab={panel} onTab={openPanel} onClose={() => setPanel(null)} memoryActions={memoryActions}
          profileStore={profileStore} onSessionProfile={selectProfile} onProfileChanged={changedProfile} />
      </>}
    </div>
  )
}
