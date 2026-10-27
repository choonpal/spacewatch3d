// Each photo shows its own capture's selected object. Opposite-capture geometry
// can locate an expected empty area, but must never be painted as an observation.
export function photoLayers(event, record, epoch, panorama = false) {
  const source = panorama ? record?.panorama?.ghosts : record?.ghosts;
  const selected = (source || []).filter(layer => !layer.key || event.object_ids.includes(layer.key));
  const own = selected.filter(layer => Number(layer.epoch) === epoch);
  return { highlights: own, bounds: own.length ? own : selected };
}

export function photoCaption(event, epoch) {
  if (!event[epoch ? 'after' : 'before']) {
    if (event.type === 'added') return '추가 전 위치';
    if (event.type === 'removed') return '소실 후 위치';
    return '대응 확인 위치';
  }
  if (event.type === 'moved') return epoch ? '도착 객체' : '출발 객체';
  return epoch ? '선택 객체 · 후' : '선택 객체 · 전';
}
