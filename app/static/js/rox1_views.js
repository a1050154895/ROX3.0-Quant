/**
 * Rox 1.0 经典界面：宏观、每周推荐、个股诊断、知识库、交易、社区、系统状态
 * 对接 3.0 API
 */
(function () {
  'use strict';

  /** 统一 token 刷新：即将过期（5 分钟内）时自动刷新，避免多标签 401 */
  function ensureTokenRefreshed() {
    var token = typeof getToken === 'function' ? getToken() : (window._rox_token || localStorage.getItem('access_token'));
    if (!token || token.indexOf('.') === -1) return Promise.resolve();
    try {
      var payload = JSON.parse(atob(token.split('.')[1]));
      var exp = payload.exp ? payload.exp * 1000 : 0;
      if (!exp || (exp - Date.now() >= 5 * 60 * 1000)) return Promise.resolve();
      return fetch('/token/refresh', { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, credentials: 'same-origin' })
        .then(function (r) {
          if (r.ok) return r.json();
          return {};
        })
        .then(function (data) {
          if (data.access_token) localStorage.setItem('access_token', data.access_token);
        })
        .catch(function () { });
    } catch (e) {
      return Promise.resolve();
    }
  }

  function apiGet(url, opts) {
    opts = opts || {};
    var headers = opts.headers || {};
    return ensureTokenRefreshed().then(function () {
      var token = typeof getToken === 'function' ? getToken() : (window._rox_token || localStorage.getItem('access_token'));
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch(url, { method: 'GET', headers: headers, credentials: 'same-origin' }).then(function (r) {
        if (r.status === 401 && typeof showAuthModal === 'function') showAuthModal();
        return r;
      });
    });
  }

  function apiPost(url, body, opts) {
    opts = opts || {};
    var headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
    return ensureTokenRefreshed().then(function () {
      var token = typeof getToken === 'function' ? getToken() : (window._rox_token || localStorage.getItem('access_token'));
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch(url, { method: 'POST', headers: headers, body: JSON.stringify(body), credentials: 'same-origin' }).then(function (r) {
        if (r.status === 401 && typeof showAuthModal === 'function') showAuthModal();
        return r;
      });
    });
  }

  // ---------- 市场看板 (热力图 + 南北资金 + 资金流 + 趋势预测) ----------
  function renderHSGTChart(domId, seriesData, name, color) {
    var dom = document.getElementById(domId);
    if (!dom || !seriesData || seriesData.length === 0 || typeof echarts === 'undefined') return;
    var ch = echarts.init(dom);
    var xData = seriesData.map(function (i) { return (i.time || '').split(' ')[0] || i.time; });
    var yData = seriesData.map(function (i) { return i.value; });
    ch.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
      grid: { top: 2, bottom: 2, left: 2, right: 2, containLabel: false },
      xAxis: { type: 'category', data: xData, show: false },
      yAxis: { type: 'value', show: false, scale: true },
      series: [{
        name: name,
        type: 'line',
        data: yData,
        showSymbol: false,
        smooth: true,
        lineStyle: { color: color, width: 1 },
        areaStyle: { color: color + '30' }
      }]
    });
    ch.resize();
  }

  var MARKET_INDEX_CODE = '000001';

  function loadMarketIndexHeader() {
    var priceEl = document.getElementById('market-index-price');
    var changeEl = document.getElementById('market-index-change');
    if (!priceEl && !changeEl) return;
    apiGet('/api/market/prediction?horizon=1')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        if (!isMarketViewActive()) return;
        var hist = data.history || {};
        var prices = hist.prices || [];
        var last = prices.length ? prices[prices.length - 1] : null;
        if (last != null && priceEl) priceEl.textContent = FormatUtils.formatPrice(last);
        if (last != null && prices.length >= 2) {
          var prev = prices[prices.length - 2];
          var ch = last - prev;
          var pct = prev ? ((ch / prev) * 100) : 0;
          if (changeEl) {
            changeEl.textContent = (ch >= 0 ? '+' : '') + ch.toFixed(2) + ' ' + FormatUtils.formatPct(pct);
            changeEl.className = 'text-sm font-mono ' + FormatUtils.getColorClass(ch);
          }
        } else if (changeEl) changeEl.textContent = '--';
      })
      .catch(function () {
        if (!isMarketViewActive()) return;
        if (priceEl) priceEl.textContent = '--';
        if (changeEl) changeEl.textContent = '--';
      });
  }

  function loadMarketFenshiChart() {
    var el = document.getElementById('market-fenshi-chart');
    if (!el || typeof echarts === 'undefined') return;

    // Add click event to enlarge
    el.style.cursor = 'pointer';
    el.onclick = function () {
      openLargeChartModal(MARKET_INDEX_CODE, 'fenshi');
    };

    apiGet('/api/market/fenshi?code=' + MARKET_INDEX_CODE)
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (d) {
        if (!isMarketViewActive()) return;
        var times = d.times || [];
        var prices = d.prices || [];
        var volumes = d.volumes || [];
        if (!times.length || !prices.length) return;
        var ch = echarts.init(el);
        ch.setOption({
          backgroundColor: 'transparent',
          tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(30,41,59,0.95)',
            borderColor: '#334155',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
            formatter: function (params) {
              if (!params || !params.length) return '';
              var idx = params[0].dataIndex;
              var t = times[idx] || '';
              var p = FormatUtils.formatPrice(prices[idx]);
              var v = FormatUtils.formatBigNumber(volumes[idx]);
              return t + '<br/>价格: ' + p + '<br/>成交量: ' + v;
            }
          },
          grid: [{ left: 50, right: 30, top: 15, bottom: 50 }, { left: 50, right: 30, top: '78%', height: '18%' }],
          xAxis: [
            { type: 'category', data: times, gridIndex: 0, axisLabel: { color: '#94a3b8', fontSize: 10 } },
            { type: 'category', data: times, gridIndex: 1, axisLabel: { show: false } }
          ],
          yAxis: [
            { type: 'value', gridIndex: 0, scale: true, splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
            { type: 'value', gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }
          ],
          series: [
            { name: '价格', type: 'line', data: prices, smooth: true, symbol: 'none', lineStyle: { color: '#38bdf8' }, areaStyle: { color: 'rgba(56,189,248,0.2)' } },
            { name: '成交量', type: 'bar', data: volumes, xAxisIndex: 1, yAxisIndex: 1, itemStyle: { color: function (p) { return (prices[p.dataIndex] >= (prices[p.dataIndex - 1] || prices[p.dataIndex])) ? '#ef4444' : '#22c55e'; } } }
          ]
        });
        ch.resize();
      })
      .catch(function () { });
  }

  function loadMarketKlineChart(period) {
    var el = document.getElementById('market-kline-chart');
    if (!el || typeof echarts === 'undefined') return;
    apiGet('/api/market/kline?code=' + MARKET_INDEX_CODE + '&period=' + (period || 'daily'))
      .then(function (r) {
        if (!r.ok) {
          return r.json().then(function (d) { throw new Error(d.error || '加载失败'); }).catch(function () { throw new Error('行情数据源暂时不可用，请稍后重试'); });
        }
        return r.json();
      })
      .then(function (d) {
        if (!isMarketViewActive()) return;
        var dates = d.dates || [];
        var ohlc = d.ohlc || [];
        if (!dates.length || !ohlc.length) {
          el.innerHTML = '<div class="h-full flex items-center justify-center text-slate-500 text-sm">K 线数据为空，请稍后重试</div>';
          return;
        }
        var ch = echarts.init(el);
        ch.setOption({
          backgroundColor: 'transparent',
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            backgroundColor: 'rgba(30,41,59,0.95)',
            borderColor: '#334155',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
            formatter: function (params) {
              if (!params || !params.length || !params[0].data) return '';
              var idx = params[0].dataIndex;
              var d = dates[idx] || '';
              var ohlcItem = ohlc[idx];
              if (!ohlcItem || ohlcItem.length < 4) return d;
              return d + '<br/>开: ' + FormatUtils.formatPrice(ohlcItem[0]) + '<br/>收: ' + FormatUtils.formatPrice(ohlcItem[1]) + '<br/>低: ' + FormatUtils.formatPrice(ohlcItem[2]) + '<br/>高: ' + FormatUtils.formatPrice(ohlcItem[3]);
            }
          },
          grid: { left: 50, right: 20, top: 15, bottom: 35 },
          xAxis: { type: 'category', data: dates, axisLabel: { color: '#94a3b8', fontSize: 10 } },
          yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
          dataZoom: [{ type: 'inside', start: 70, end: 100 }, { type: 'slider', bottom: 0, height: 18 }],
          series: [
            { type: 'candlestick', data: ohlc, itemStyle: { color: '#ef4444', color0: '#22c55e', borderColor: '#ef4444', borderColor0: '#22c55e' } }
          ]
        });
        ch.resize();
        if (d.fallback && el.parentElement && !el.parentElement.querySelector('.kline-fallback-hint')) {
          var hint = document.createElement('span');
          hint.className = 'kline-fallback-hint block text-xxs text-amber-500/80 mt-1';
          hint.textContent = '数据为模拟（行情源暂时不可用）';
          el.parentElement.appendChild(hint);
        }
      })
      .catch(function (e) {
        if (!el) return;
        el.innerHTML = '<div class="h-full flex items-center justify-center text-slate-500 text-sm px-2">' + (e.message || '行情数据源暂时不可用，请稍后重试') + '</div>';
      });
  }

  function isMarketViewActive() {
    var el = document.getElementById('view-market');
    return el && el.classList.contains('view-active');
  }

  function loadMarketDashboard() {
    if (!isMarketViewActive()) return;
    var northLive = document.getElementById('north-live');
    var southLive = document.getElementById('south-live');
    var sectorHeat = document.getElementById('sector-heatmap');
    var fundFlowList = document.getElementById('sector-fund-flow-list');
    var predConf = document.getElementById('pred-confidence');
    var predLogic = document.getElementById('pred-logic');
    var predSupport = document.getElementById('pred-support');
    var predPressure = document.getElementById('pred-pressure');
    var predChartEl = document.getElementById('pred-chart');
    var predChartWrap = document.getElementById('pred-chart-wrap');
    var aiSummaryEl = document.getElementById('market-ai-summary');
    var indexBarEl = document.getElementById('market-index-bar');
    var newsListEl = document.getElementById('market-news-list');

    loadMarketIndexHeader();

    // 每日 AI 市场总结
    if (aiSummaryEl) {
      apiGet('/api/market/briefing')
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (data) {
          if (!isMarketViewActive()) return;
          if (aiSummaryEl) aiSummaryEl.textContent = data.briefing || '暂无今日总结，请稍后刷新。';
        })
        .catch(function () {
          if (aiSummaryEl) aiSummaryEl.textContent = 'AI 总结加载失败，请稍后重试。';
        });
    }

    // 资讯栏目
    if (newsListEl) {
      apiGet('/api/market/news')
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (data) {
          if (!isMarketViewActive()) return;
          var list = data.news || [];
          if (list.length === 0) newsListEl.innerHTML = '<span class="text-slate-500">暂无资讯</span>';
          else {
            newsListEl.innerHTML = list.slice(0, 15).map(function (n) {
              var title = (n.title || n.name || '').substring(0, 36);
              var date = n.date || n.time || '';
              var url = (n.url || '').trim() || '#';
              if (url === '#') url = 'https://www.baidu.com/s?wd=' + encodeURIComponent(n.title || n.name || '');
              return '<a href="' + url.replace(/"/g, '&quot;') + '" target="_blank" rel="noopener noreferrer" class="flex justify-between gap-2 py-0.5 border-b border-slate-700/50 hover:bg-slate-700/30 rounded px-1 -mx-1 text-slate-400 hover:text-slate-200 no-underline"><span class="truncate">' + title + (title.length >= 36 ? '…' : '') + '</span><span class="text-slate-500 shrink-0">' + date + '</span></a>';
            }).join('');
          }
        })
        .catch(function () {
          if (newsListEl) newsListEl.innerHTML = '<span class="text-slate-500">资讯加载失败</span>';
        });
    }

    // 顶部主要指数（上证、深证、创业板、沪深300）+ 底部指数条
    var indexTopEl = document.getElementById('market-index-top');
    if (indexBarEl || indexTopEl) {
      apiGet('/api/market/indices')
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (data) {
          if (!isMarketViewActive()) return;
          var list = data.indices || [];
          var topNames = ['上证指数', '深证成指', '创业板指', '沪深300'];
          if (indexTopEl) {
            if (list.length === 0) indexTopEl.innerHTML = '<span class="text-slate-500">指数加载中…</span>';
            else {
              var topList = topNames.map(function (n) { return list.find(function (i) { return i.name === n; }); }).filter(Boolean);
              if (topList.length === 0) topList = list.slice(0, 4);
              indexTopEl.innerHTML = topList.map(function (i) {
                var p = FormatUtils.formatPrice(i.price);
                var pct = Number(i.pct) || 0;
                var cls = FormatUtils.getColorClass(pct);
                return '<div class="flex flex-col"><span class="text-slate-400 text-xs">' + (i.name || '') + '</span><span class="text-slate-200 font-mono">' + p + ' <span class="' + cls + '">' + FormatUtils.formatPct(pct) + '</span></span></div>';
              }).join('');
            }
          }
          if (indexBarEl) {
            if (list.length === 0) indexBarEl.innerHTML = '<span class="text-slate-500">暂无指数数据</span>';
            else {
              indexBarEl.innerHTML = list.map(function (i) {
                var p = FormatUtils.formatPrice(i.price);
                var pct = Number(i.pct) || 0;
                var cls = FormatUtils.getColorClass(pct);
                return '<span class="whitespace-nowrap">' + (i.name || '') + ' <span class="font-semibold">' + p + '</span> <span class="' + cls + '">' + FormatUtils.formatPct(pct) + '</span></span>';
              }).join('');
            }
          }
        })
        .catch(function () {
          if (indexTopEl) indexTopEl.innerHTML = '<span class="text-slate-500">指数加载失败</span>';
          if (indexBarEl) indexBarEl.innerHTML = '<span class="text-slate-500">指数加载失败</span>';
        });
    }
    loadMarketFenshiChart();
    loadMarketKlineChart('daily');

    document.querySelectorAll('.market-fenshi-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.market-fenshi-tab').forEach(function (b) { b.classList.remove('bg-sky-600', 'text-white'); b.classList.add('text-slate-400'); });
        btn.classList.add('bg-sky-600', 'text-white'); btn.classList.remove('text-slate-400');
        loadMarketFenshiChart();
      });
    });
    document.querySelectorAll('.market-kline-tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.market-kline-tab').forEach(function (b) { b.classList.remove('bg-sky-600', 'text-white'); b.classList.add('text-slate-400'); });
        btn.classList.add('bg-sky-600', 'text-white'); btn.classList.remove('text-slate-400');
        var p = btn.getAttribute('data-p') || 'daily';
        loadMarketKlineChart(p);
      });
    });

    // 南北资金
    apiGet('/api/market/hsgt/realtime?period=daily')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        if (!isMarketViewActive()) return;
        var north = data.north || [];
        var south = data.south || [];
        if (north.length) {
          var lastNorth = north[north.length - 1].value;
          if (northLive) {
            northLive.textContent = FormatUtils.formatBigNumber(lastNorth);
            northLive.className = 'text-lg font-mono font-bold ' + FormatUtils.getColorClass(lastNorth);
          }
          renderHSGTChart('hsgt-chart-north', north, '北向', '#ef4444');
        }
        if (south.length) {
          var lastSouth = south[south.length - 1].value;
          if (southLive) {
            southLive.textContent = FormatUtils.formatBigNumber(lastSouth);
            southLive.className = 'text-lg font-mono font-bold ' + FormatUtils.getColorClass(lastSouth);
          }
          renderHSGTChart('hsgt-chart-south', south, '南向', '#22c55e');
        }
      })
      .catch(function () {
        if (northLive) northLive.textContent = '--';
        if (southLive) southLive.textContent = '--';
      });

    // 板块热力 + 资金流向
    apiGet('/api/market/sector-fund-flow')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        if (!isMarketViewActive()) return;
        var list = data.data || [];
        if (sectorHeat) {
          if (list.length === 0) sectorHeat.innerHTML = '<span class="text-slate-500 text-sm">暂无数据</span>';
          else {
            sectorHeat.innerHTML = list.slice(0, 20).map(function (s) {
              var pct = Number(s.pct) || 0;
              var cls = FormatUtils.getColorClass(pct);
              // Use bg opacity based on pct magnitude could be cool, but let's stick to simple first
              var bgCls = pct >= 0 ? 'bg-red-500/20 border-red-500/40' : 'bg-green-500/20 border-green-500/40';
              return '<span class="px-2 py-0.5 rounded text-xs border ' + cls + ' ' + bgCls + '">' + (s.name || '') + ' ' + FormatUtils.formatPct(pct) + '</span>';
            }).join('');
          }
        }
        if (fundFlowList) {
          if (list.length === 0) fundFlowList.innerHTML = '<span class="text-slate-500">暂无数据</span>';
          else {
            fundFlowList.innerHTML = list.slice(0, 15).map(function (s) {
              var val = Number(s.value) || 0;
              var valStr = FormatUtils.formatBigNumber(val);
              var pct = Number(s.pct) || 0;
              var pctCls = FormatUtils.getColorClass(pct);
              return '<div class="flex justify-between py-0.5 border-b border-slate-700/50"><span>' + (s.name || '') + '</span><span class="font-mono ' + pctCls + '">' + valStr + ' ' + FormatUtils.formatPct(pct) + '</span></div>';
            }).join('');
          }
        }
      })
      .catch(function () {
        if (sectorHeat) sectorHeat.innerHTML = '<span class="text-slate-500 text-sm">加载失败</span>';
        if (fundFlowList) fundFlowList.innerHTML = '<span class="text-slate-500">加载失败</span>';
      });

    // 趋势预测：历史蓝色实线，预测黄色虚线；点击图表可放大
    apiGet('/api/market/prediction?horizon=5')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        if (!isMarketViewActive()) return;
        if (predConf) predConf.textContent = (data.confidence != null ? data.confidence : '--') + '%';
        if (predLogic) predLogic.textContent = data.logic || '--';
        if (predSupport) predSupport.textContent = data.support || '--';
        if (predPressure) predPressure.textContent = data.pressure || '--';

        // Add click to enlarge
        if (predChartWrap) {
          predChartWrap.onclick = function () {
            openLargeChartModal('market_prediction', 'prediction');
          };
        }

        var history = data.history || {};
        var histDates = history.dates || [];
        var histPrices = history.prices || [];
        var predDates = data.dates || [];
        var predPrices = data.prediction || [];
        var allDates = histDates.concat(predDates);
        var option = {
          backgroundColor: 'transparent',
          tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(30,41,59,0.95)',
            borderColor: '#334155',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
            formatter: function (params) {
              if (!params || !params.length) return '';
              var res = '<div class="font-bold border-b border-slate-600 pb-1 mb-1">' + params[0].name + '</div>';
              params.forEach(function (p) {
                if (p.value !== null && p.value !== undefined) {
                  var color = p.seriesName === '预测' ? '#facc15' : '#38bdf8';
                  res += '<div class="flex justify-between gap-4"><span style="color:' + color + '">● ' + p.seriesName + '</span><span class="font-mono">' + p.value + '</span></div>';
                }
              });
              return res;
            }
          },
          grid: { left: 40, right: 15, top: 10, bottom: 25, containLabel: false },
          xAxis: {
            type: 'category',
            data: allDates,
            axisLabel: {
              color: '#94a3b8',
              fontSize: 10,
              interval: 'auto', // Let echarts decide, or use 0 if we want all
              formatter: function (value) {
                // Return simple date if it's too long? No, user wants detail.
                return value;
              }
            }
          },
          yAxis: { type: 'value', scale: true, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: '#334155' } } },
          series: []
        };
        if (histDates.length > 0 && histPrices.length > 0) {
          var histData = histPrices.slice();
          for (var i = 0; i < predDates.length; i++) histData.push(null);
          option.series.push({
            name: '实际',
            type: 'line',
            data: histData,
            smooth: true,
            symbol: 'none',
            lineStyle: { color: '#38bdf8', width: 2, type: 'solid' },
            areaStyle: { color: 'rgba(56,189,248,0.15)' },
            connectNulls: false
          });
        }
        if (predDates.length > 0 && predPrices.length > 0) {
          var predData = [];
          for (var j = 0; j < histDates.length - 1; j++) predData.push(null);
          if (histPrices.length > 0) predData.push(histPrices[histPrices.length - 1]);
          for (var k = 0; k < predPrices.length; k++) predData.push(predPrices[k]);
          option.series.push({
            name: '预测',
            type: 'line',
            data: predData,
            smooth: true,
            symbol: 'none',
            lineStyle: { color: '#facc15', width: 2, type: 'dashed' },
            connectNulls: true
          });
        }
        if (option.series.length === 0 && histDates.length > 0 && histPrices.length > 0) {
          option.series.push({
            name: '实际',
            type: 'line',
            data: histPrices,
            smooth: true,
            symbol: 'none',
            lineStyle: { color: '#38bdf8', width: 2, type: 'solid' },
            areaStyle: { color: 'rgba(56,189,248,0.15)' }
          });
        }
        window._predChartOption = option;
        if (predChartEl && typeof echarts !== 'undefined' && option.series.length > 0) {
          var ch = echarts.init(predChartEl);
          ch.setOption(option);
          ch.resize();
        }
        if (predChartWrap) {
          predChartWrap.onclick = function () {
            if (window._predChartOption && typeof openPredChartModal === 'function') openPredChartModal(window._predChartOption);
          };
        }
      })
      .catch(function () {
        if (predLogic) predLogic.textContent = '预测加载失败，请稍后重试。';
      });
  }

  function openPredChartModal(option) {
    var modal = document.getElementById('pred-chart-modal');
    var body = document.getElementById('pred-chart-modal-body');
    if (!modal || !body) return;
    body.innerHTML = '<div id="pred-chart-modal-chart" class="w-full h-[320px]"></div>';
    modal.classList.remove('hidden');
    modal.style.display = 'flex';
    setTimeout(function () {
      var el = document.getElementById('pred-chart-modal-chart');
      if (el && typeof echarts !== 'undefined' && option) {
        var ch = echarts.init(el);
        ch.setOption(option);
        ch.resize();
      }
    }, 50);
  }

  function closePredChartModal() {
    var modal = document.getElementById('pred-chart-modal');
    if (modal) {
      modal.classList.add('hidden');
      modal.style.display = 'none';
    }
  }
  window.closePredChartModal = closePredChartModal;
  window.openPredChartModal = openPredChartModal;

  // ---------- 宏观仪表盘 ----------
  function loadMacro() {
    var el = document.getElementById('view-macro');
    if (!el) return;
    var inner = el.querySelector('.rox1-macro-inner');
    if (!inner) return;
    inner.innerHTML = '<p class="text-slate-400 text-sm">加载中…</p>';
    apiGet('/api/market/macro')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        var labels = {
          bond_10y: '10年国债(%)',
          pmi_mfg: '制造业PMI',
          pmi_svc: '服务业PMI',
          cpi: 'CPI(%)',
          ppi: 'PPI(%)',
          shibor: 'Shibor(%)',
          gdp_growth: 'GDP增速(%)',
          m2: 'M2同比(%)',
          soc_fin: '社融(%)',
          usd_cny: '美元/人民币',
          crb: 'CRB指数'
        };
        var html = '<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">';
        for (var k in labels) {
          if (!labels.hasOwnProperty(k)) continue;
          var v = data[k];
          if (v === undefined) v = '--';
          else if (typeof v === 'number') v = v.toFixed(2);
          html += '<div class="glass p-4 rounded-lg border border-slate-700/50"><div class="text-xs text-slate-500 mb-1">' + labels[k] + '</div><div class="text-xl font-mono text-sky-300">' + v + '</div></div>';
        }
        html += '</div>';
        inner.innerHTML = html;
      })
      .catch(function () {
        inner.innerHTML = '<p class="text-slate-400 text-sm">加载失败，请稍后重试。</p>';
      });
  }

  // ---------- 每周推荐 ----------
  function loadWeekly() {
    var el = document.getElementById('view-weekly');
    if (!el) return;
    var cardsEl = el.querySelector('.rox1-weekly-cards');
    var chartEl = el.querySelector('#rox1-weekly-334-chart');
    if (!cardsEl) return;
    cardsEl.innerHTML = '<p class="text-slate-400 text-sm">加载中…</p>';
    apiGet('/api/market/weekly')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        var briefEl = el.querySelector('#rox1-weekly-macro-brief');
        var posEl = el.querySelector('#rox1-weekly-position');
        var sectorEl = el.querySelector('#rox1-weekly-sector');
        if (briefEl) briefEl.textContent = data.macro_brief || '';
        if (posEl) posEl.textContent = data.position_suggestion || '';
        if (sectorEl) sectorEl.textContent = data.sector_style || '';

        var items = data.items || [];
        var html = '';
        try {
          items.forEach(function (item) {
            html += '<div class="glass p-4 rounded-xl border border-slate-700/50">';
            html += '<div class="flex justify-between items-center mb-2"><span class="font-bold text-slate-200">' + (item.name || '') + '</span><span class="text-xs font-mono text-slate-500">' + (item.code || '') + '</span></div>';
            html += '<p class="text-sm text-slate-400 mb-2">' + (item.reason || '') + '</p>';
            if (item.tags && item.tags.length) {
              html += '<div class="flex flex-wrap gap-1 mb-2">';
              item.tags.forEach(function (t) { html += '<span class="px-1.5 py-0.5 rounded text-xxs bg-slate-700/80 text-sky-300">' + (t || '') + '</span>'; });
              html += '</div>';
            }
            if (item.score_breakdown && typeof item.score_breakdown === 'object') {
              var sb = item.score_breakdown;
              html += '<div class="text-xxs text-slate-500 mb-2">技术 ' + (sb.tech != null ? sb.tech : '--') + ' / 资金 ' + (sb.fund != null ? sb.fund : '--') + ' / 基本面 ' + (sb.fundamental != null ? sb.fundamental : '--') + '</div>';
            }
            html += '<div class="flex gap-4 text-xs"><span class="text-up">目标 ' + (item.target ? FormatUtils.formatPrice(item.target) : '--') + '</span><span class="text-down">止损 ' + (item.stop ? FormatUtils.formatPrice(item.stop) : '--') + '</span><span class="text-amber-400">评分 ' + (item.score || '--') + '</span></div>';
            html += '</div>';
          });
          cardsEl.innerHTML = html || '<p class="text-slate-400 text-sm">暂无推荐</p>';
        } catch (e) {
          console.error("Weekly items render failed:", e);
          cardsEl.innerHTML = '<p class="text-slate-400 text-sm">推荐加载异常</p>';
        }

        try {
          var s334 = data.strategy_334;
          if (s334 && chartEl && typeof echarts !== 'undefined') {
            var colors = ['#38bdf8', '#facc15', '#34d399'];
            var barColors = (s334.data || []).map(function (_, i) { return colors[i % colors.length]; });
            var ch = echarts.init(chartEl);
            ch.setOption({
              tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155', textStyle: { color: '#e2e8f0', fontSize: 11 } },
              grid: { left: 10, right: 10, top: 20, bottom: 10, containLabel: true },
              xAxis: { type: 'category', data: s334.labels || [], axisLabel: { color: '#94a3b8' } },
              yAxis: { type: 'value', max: 100, axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: '#334155' } } },
              series: [{ type: 'bar', barWidth: '60%', data: (s334.data || []).map(function (v, i) { return { value: v, itemStyle: { color: barColors[i] } }; }) }]
            });
            window.addEventListener('resize', function () { ch.resize(); });
          }
        } catch (e) {
          console.error("334 Chart render failed:", e);
          if (chartEl) chartEl.innerHTML = '<p class="text-xs text-slate-500">图表加载失败</p>';
        }
      })
      .catch(function () {
        cardsEl.innerHTML = '<p class="text-slate-400 text-sm">加载失败。</p>';
      });
  }

  function fillDiagnosisFormCard(data, rs, diag) {
    var cur = document.getElementById('diag-f-current');
    var target = document.getElementById('diag-f-target');
    var stop = document.getElementById('diag-f-stop');
    var chips = document.getElementById('diag-f-chips');
    var vol = document.getElementById('diag-f-volume');
    var resonance = document.getElementById('diag-f-resonance');
    if (data) {
      if (cur) cur.textContent = (data.p_now != null && data.p_now !== '') ? FormatUtils.formatPrice(data.p_now) : '--';
      if (chips) chips.textContent = (data.chips_ratio != null) ? FormatUtils.formatPct(data.chips_ratio * 100) : '--';
      if (vol) {
        var v = data.volume_increase;
        if (v != null && v !== '') {
          // Ensure it has '倍' if it's a number
          if (!isNaN(v)) vol.textContent = v + '倍';
          else vol.textContent = v;
        } else {
          vol.textContent = '--';
        }
      }
    }
    if (rs) {
      if (target && (rs.tomorrow_breakout != null || rs.today_resistance != null)) target.textContent = rs.tomorrow_breakout != null ? FormatUtils.formatPrice(rs.tomorrow_breakout) : FormatUtils.formatPrice(rs.today_resistance);
      if (stop && rs.today_support != null) stop.textContent = FormatUtils.formatPrice(rs.today_support);
    }
    if (diag) {
      if (resonance) resonance.textContent = diag.overall_score != null ? diag.overall_score + ' 分' : '--';

      // Use prediction data if available (New in 3.0)
      if (diag.details && diag.details.prediction) {
        if (target) target.textContent = FormatUtils.formatPrice(diag.details.prediction.target_price);
        if (stop) stop.textContent = FormatUtils.formatPrice(diag.details.prediction.stop_loss);
      }
      // Fallback to technical data
      else if (diag.details && diag.details.technical) {
        if (target && target.textContent === '--' && diag.details.technical.resistance) target.textContent = FormatUtils.formatPrice(diag.details.technical.resistance);
        if (stop && stop.textContent === '--' && diag.details.technical.support) stop.textContent = FormatUtils.formatPrice(diag.details.technical.support);
      }

      // Fill Volume Increase if available in diagnosis details
      if (vol && vol.textContent === '--' && diag.details && diag.details.volume && diag.details.volume.ratio) {
        vol.textContent = diag.details.volume.ratio + '倍';
      }
    }
  }
  function updateDiagnosisFormFromApi(diag) {
    fillDiagnosisFormCard(null, null, diag);
  }
  function initDiagnosisSidebar() {
    var btn = document.getElementById('diag-btn');
    if (btn && !btn._bound) {
      btn._bound = true;
      btn.addEventListener('click', runDiagnosis);
    }
    var toTradeBtn = document.getElementById('diag-to-trade-btn');
    if (toTradeBtn && !toTradeBtn._bound) {
      toTradeBtn._bound = true;
      toTradeBtn.addEventListener('click', function () {
        var sym = document.getElementById('trade-symbol');
        var name = document.getElementById('trade-name');
        var price = document.getElementById('trade-price');
        var stopInput = document.getElementById('trade-stop-loss');
        var targetInput = document.getElementById('trade-target-price');
        if (sym) sym.value = window._lastDiagnosisSymbol || '';
        if (name) name.value = window._lastDiagnosisName || '';
        var d = window._lastDiagnosisData;
        if (d && d.p_now != null) { if (price) price.value = d.p_now; }
        var tEl = document.getElementById('diag-f-target');
        var sEl = document.getElementById('diag-f-stop');
        if (tEl && tEl.textContent !== '--' && targetInput) targetInput.value = tEl.textContent;
        if (sEl && sEl.textContent !== '--' && stopInput) stopInput.value = sEl.textContent;
        if (typeof showToast === 'function') showToast('已带入交易表单，请填写数量后下单');
        if (typeof switchMode === 'function') switchMode('trading');
      });
    }
  }

  function runDiagnosis() {
    var input = document.getElementById('diag-input');
    var resultEl = document.getElementById('diag-result');
    if (!input) return;
    var q = (input.value || '').trim();
    if (!q) {
      if (resultEl) { resultEl.textContent = '请输入代码或名称'; resultEl.classList.remove('hidden'); }
      return;
    }
    if (resultEl) { resultEl.textContent = '查询中…'; resultEl.classList.remove('hidden'); }
    apiPost('/api/market/fetch-realtime', { stock_name: q })
      .then(function (r) {
        if (!r.ok) {
          return r.json().then(function (j) {
            var msg = j && (j.error || j.detail);
            if (typeof msg === 'string') throw new Error(msg);
            if (r.status === 503) throw new Error('行情服务暂不可用，请稍后重试或直接输入6位股票代码（如 600519）');
            if (r.status === 404) throw new Error('未找到该股票，请检查代码或名称，或尝试输入6位代码');
            throw new Error(msg || '查询失败');
          }).catch(function () {
            if (r.status === 503) throw new Error('行情服务暂不可用，请稍后重试或直接输入6位股票代码');
            if (r.status === 404) throw new Error('未找到该股票，请尝试输入6位代码');
            throw new Error('查询失败，请稍后重试');
          });
        }
        return r.json();
      })
      .then(function (data) {
        var pStr = (data.p_now != null && data.p_now !== '') ? FormatUtils.formatPrice(data.p_now) : '--';
        var vStr = (data.volume_increase != null && data.volume_increase !== '') ? data.volume_increase : '--';
        if (vStr !== '--' && !isNaN(parseFloat(vStr))) vStr += '倍';
        var chipsStr = (data.chips_ratio != null ? FormatUtils.formatPct(data.chips_ratio * 100) : '--');
        var t = '现价: ' + pStr + ' | 量能: ' + vStr + ' | 获利盘: ' + chipsStr;
        if (data.fundamentals) {
          t += ' | PE ' + (data.fundamentals.pe || '--');
          t += ' | 市值 ' + (data.fundamentals.mv ? FormatUtils.formatBigNumber(data.fundamentals.mv) : '--');
        }
        if (resultEl) { resultEl.textContent = t; resultEl.classList.remove('hidden'); }
        window._lastDiagnosisSymbol = data.code || '';
        window._lastDiagnosisName = data.name || '';
        window._lastDiagnosisData = data;
        fillDiagnosisFormCard(data);
        var viewResult = document.getElementById('diagnosis-result-main');
        var code = data.code;
        if (!viewResult) return;
        viewResult.classList.remove('hidden');
        viewResult.innerHTML = '<div class="glass p-4 rounded-xl"><p class="text-slate-200 font-mono text-sm">' + (data.name || '') + ' ' + (data.code || '') + '</p><p class="text-slate-400 text-sm mt-2">' + t + '</p><p class="text-xxs text-slate-500 mt-2" id="diag-rs-placeholder">阻力/支撑加载中…</p><p class="text-xxs text-slate-500 mt-2" id="diag-summary-placeholder">诊断结论加载中…</p></div>';
        if (code) {
          apiGet('/api/stock/resistance-support?code=' + encodeURIComponent(code))
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (rs) {
              var ph = document.getElementById('diag-rs-placeholder');
              if (!ph) return;
              if (rs && typeof rs === 'object') {
                ph.innerHTML = '今日阻力 ' + (rs.today_resistance != null ? FormatUtils.formatPrice(rs.today_resistance) : '--') + ' 元 / 今日支撑 ' + (rs.today_support != null ? FormatUtils.formatPrice(rs.today_support) : '--') + ' 元；明日突破 ' + (rs.tomorrow_breakout != null ? FormatUtils.formatPrice(rs.tomorrow_breakout) : '--') + ' / 明日阻力 ' + (rs.tomorrow_resistance != null ? FormatUtils.formatPrice(rs.tomorrow_resistance) : '--') + ' / 明日支撑 ' + (rs.tomorrow_support != null ? FormatUtils.formatPrice(rs.tomorrow_support) : '--') + ' / 明日反转 ' + (rs.tomorrow_reversal != null ? FormatUtils.formatPrice(rs.tomorrow_reversal) : '--') + ' 元。';
              } else {
                ph.textContent = '阻力/支撑暂不可用';
              }
              fillDiagnosisFormCard(window._lastDiagnosisData, rs, null);
            })
            .catch(function () { var ph = document.getElementById('diag-rs-placeholder'); if (ph) ph.textContent = '阻力/支撑加载失败'; });
          apiGet('/api/stock/diagnose?code=' + encodeURIComponent(code))
            .then(function (r) {
              if (!r.ok) return null;
              return r.json();
            })
            .then(function (diag) {
              var ph = document.getElementById('diag-summary-placeholder');
              if (!ph) return;
              if (diag && typeof diag === 'object') {
                var score = diag.overall_score != null ? diag.overall_score : '--';
                var summary = (diag.summary || '').trim() || '暂无诊断结论';
                var scores = diag.scores || {};
                var details = diag.details || {};
                var tech = details.technical || {};
                var fund = details.fundamental || {};
                var flow = details.fund_flow || {};
                var vol = details.volume || {};
                var pred = details.prediction || {};

                var html = '<div class="space-y-2">';

                // 1. 综合得分与总结
                html += '<div class="border-b border-slate-700/50 pb-2 mb-2">';
                html += '<span class="text-amber-400 font-semibold text-lg">综合得分 ' + score + ' 分</span>';
                html += '<p class="text-slate-300 text-sm mt-1 leading-relaxed">' + summary + '</p>';
                html += '<div class="flex gap-4 mt-2 text-xs text-slate-400">';
                html += '<span>技术 ' + (scores.technical != null ? scores.technical : '--') + '</span>';
                html += '<span>基本面 ' + (scores.fundamental != null ? scores.fundamental : '--') + '</span>';
                html += '<span>资金 ' + (scores.fund_flow != null ? scores.fund_flow : '--') + '</span>';
                html += '</div></div>';

                // 2. 预测与操作建议 (目标/止损/T+0)
                if (pred.target_price || pred.stop_loss || pred.t_plus_zero) {
                  html += '<div class="bg-slate-700/30 p-2 rounded border border-slate-600/30">';
                  html += '<h4 class="text-sky-400 text-xs font-bold mb-1">操作建议</h4>';
                  html += '<div class="grid grid-cols-2 gap-2 text-xs mb-1">';
                  html += '<div>目标价: <span class="text-red-400 font-mono">' + (pred.target_price ? FormatUtils.formatPrice(pred.target_price) : '--') + '</span></div>';
                  html += '<div>止损价: <span class="text-green-400 font-mono">' + (pred.stop_loss ? FormatUtils.formatPrice(pred.stop_loss) : '--') + '</span></div>';
                  html += '</div>';
                  if (pred.t_plus_zero) {
                    html += '<div class="text-xs text-slate-400 mt-1 border-t border-slate-600/30 pt-1">';
                    html += '<span class="text-amber-500">T+0参考:</span> ';
                    html += '买入 <span class="font-mono text-slate-200">' + (pred.t_plus_zero.buy || '--') + '</span> / ';
                    html += '卖出 <span class="font-mono text-slate-200">' + (pred.t_plus_zero.sell || '--') + '</span>';
                    if (pred.t_plus_zero.note) html += '<div class="text-[10px] text-slate-500 mt-0.5">' + pred.t_plus_zero.note + '</div>';
                    html += '</div>';
                  }
                  html += '</div>';
                }

                // 3. 量能分析
                if (vol.summary) {
                  html += '<div class="text-xs text-slate-400"><span class="text-sky-400 font-bold">量能:</span> ' + vol.summary + '</div>';
                }

                // 4. 资金动向
                if (flow.summary) {
                  html += '<div class="text-xs text-slate-400"><span class="text-sky-400 font-bold">资金:</span> ' + flow.summary + '</div>';
                }

                // 5. 基本面 & 财务
                if (fund.summary) {
                  html += '<div class="text-xs text-slate-400"><span class="text-sky-400 font-bold">基本面:</span> ' + fund.summary + '</div>';
                  if (fund.metrics) {
                    var m = fund.metrics;
                    html += '<div class="flex gap-3 text-[10px] text-slate-500 mt-0.5 font-mono">';
                    if (m.pe_ratio) html += 'PE: ' + m.pe_ratio.toFixed(1);
                    if (m.pb_ratio) html += 'PB: ' + m.pb_ratio.toFixed(1);
                    if (m.roe) html += 'ROE: ' + m.roe.toFixed(1) + '%';
                    html += '</div>';
                  }
                }

                // 6. 相关资讯
                var newsList = details.news || [];
                if (newsList.length > 0) {
                  html += '<div class="mt-2 border-t border-slate-700/50 pt-2">';
                  html += '<h4 class="text-sky-400 text-xs font-bold mb-1">相关资讯</h4>';
                  html += '<ul class="space-y-1">';
                  newsList.slice(0, 3).forEach(function (n) {
                    var title = (n.title || '').substring(0, 20) + (n.title && n.title.length > 20 ? '...' : '');
                    var url = n.url || '#';
                    html += '<li><a href="' + url + '" target="_blank" class="text-[10px] text-slate-400 hover:text-sky-300 flex justify-between"><span>' + title + '</span><span class="text-slate-600">' + (n.time || '').split(' ')[0] + '</span></a></li>';
                  });
                  html += '</ul></div>';
                }

                html += '</div>'; // End container

                ph.innerHTML = html;
                updateDiagnosisFormFromApi(diag);
              } else {
                ph.textContent = '诊断接口暂不可用，仅显示行情与阻力支撑。';
              }
            })
            .catch(function () { var ph = document.getElementById('diag-summary-placeholder'); if (ph) ph.textContent = '诊断加载失败，仅显示行情。'; });
        }
      })
      .catch(function (e) {
        if (resultEl) {
          resultEl.textContent = e.message || '查询失败';
          resultEl.classList.remove('hidden');
        }
      });
  }

  // ---------- 选股器（寻龙诀 + AI 总结） ----------
  function loadScreener() {
    var runBtn = document.getElementById('screener-run-btn');
    var statusEl = document.getElementById('screener-status');
    var resultsEl = document.getElementById('screener-results');
    var summaryEl = document.getElementById('screener-ai-summary');
    var aiCheck = document.getElementById('screener-ai');
    if (!runBtn || !resultsEl) return;
    runBtn.onclick = function () {
      statusEl.textContent = '选股中…';
      resultsEl.innerHTML = '<p class="text-slate-400 text-sm">选股中…</p>';
      summaryEl.classList.add('hidden');
      var useAi = aiCheck && aiCheck.checked;
      if (useAi) {
        apiPost('/api/strategy/screen-with-ai', { screen_type: 'xunlongjue', max_results: 30 })
          .then(function (r) { return r.ok ? r.json() : {}; })
          .then(function (data) {
            statusEl.textContent = '';
            var items = data.items || [];
            if (data.ai_summary) {
              summaryEl.textContent = data.ai_summary;
              summaryEl.classList.remove('hidden');
            }
            if (items.length === 0) {
              resultsEl.innerHTML = '<p class="text-slate-500 text-sm">未筛出标的</p>';
              return;
            }
            var html = '<ul class="space-y-2">';
            items.forEach(function (x) {
              html += '<li class="glass p-2 rounded border border-slate-700/50 flex justify-between items-center"><span class="font-mono text-sky-300">' + (x.code || '') + '</span><span class="text-slate-300">' + (x.name || '') + '</span><span class="text-xxs text-slate-500">' + (x.reason || '') + '</span></li>';
            });
            html += '</ul>';
            resultsEl.innerHTML = html;
          })
          .catch(function () { statusEl.textContent = ''; resultsEl.innerHTML = '<p class="text-red-400 text-sm">选股失败</p>'; });
      } else {
        apiGet('/api/strategy/screen?max_codes=30')
          .then(function (r) { return r.ok ? r.json() : {}; })
          .then(function (data) {
            statusEl.textContent = '';
            var items = data.items || data.results || [];
            if (items.length === 0) {
              resultsEl.innerHTML = '<p class="text-slate-500 text-sm">未筛出标的</p>';
              return;
            }
            var html = '<ul class="space-y-2">';
            items.forEach(function (x) {
              var code = x.code || x.symbol || '';
              var name = x.name || '';
              var reason = x.reason || x.remark || '';
              html += '<li class="glass p-2 rounded border border-slate-700/50 flex justify-between items-center"><span class="font-mono text-sky-300">' + code + '</span><span class="text-slate-300">' + name + '</span><span class="text-xxs text-slate-500">' + reason + '</span></li>';
            });
            html += '</ul>';
            resultsEl.innerHTML = html;
          })
          .catch(function () { statusEl.textContent = ''; resultsEl.innerHTML = '<p class="text-red-400 text-sm">选股失败</p>'; });
      }
    };
  }

  // ---------- 知识库（搜索模式已在页面内） ----------
  function initKbSidebar() {
    // kb-mode 已在 view-kb 内，无需注入
  }

  function loadKbSearch(query, mode) {
    var listEl = document.getElementById('kb-results-list');
    if (!listEl) return;
    if (!query || !query.trim()) {
      listEl.innerHTML = '<p class="text-slate-500 text-sm">输入关键词后搜索</p>';
      return;
    }
    listEl.innerHTML = '<p class="text-slate-400 text-sm">搜索中…</p>';
    fetch('/api/kb/search?query=' + encodeURIComponent(query.trim()) + '&mode=' + (mode || 'mixed'), { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (arr) {
        if (!Array.isArray(arr)) arr = [];
        if (arr.length === 0) {
          var msg = '无结果';
          if (mode === 'web' || mode === 'mixed') msg += '。若为联网/混合模式，可能因网络原因无法访问，请使用「仅本地」或稍后重试。';
          listEl.innerHTML = '<p class="text-slate-500 text-sm">' + msg + '</p>';
          return;
        }
        var html = '<ul class="space-y-3">';
        arr.forEach(function (item) {
          html += '<li class="glass p-3 rounded-lg border border-slate-700/50">';
          html += '<div class="font-medium text-slate-200 text-sm">' + (item.title || '') + '</div>';
          html += '<div class="text-xs text-slate-400 mt-1">' + (item.snippet || '') + '</div>';
          html += '<div class="text-xxs text-slate-500 mt-1">' + (item.source || '') + '</div></li>';
        });
        html += '</ul>';
        listEl.innerHTML = html;
      })
      .catch(function () {
        listEl.innerHTML = '<p class="text-red-400 text-sm">搜索失败</p>';
      });
  }

  // ---------- 交易系统 ----------
  function loadTrading() {
    var accountsEl = document.getElementById('trading-accounts');
    var tradesEl = document.getElementById('trading-trades-list');
    var accountSelect = document.getElementById('trade-account');
    var submitBtn = document.getElementById('trading-submit-btn');

    function updateTradeButtonText() {
      if (submitBtn) submitBtn.textContent = (accountSelect && accountSelect.value === 'real') ? '下单（需对接券商）' : '记录交易';
    }
    if (accountSelect) accountSelect.addEventListener('change', updateTradeButtonText);
    updateTradeButtonText();

    var fillFromDiag = document.getElementById('trade-fill-from-diagnosis');
    if (fillFromDiag) {
      fillFromDiag.addEventListener('click', function () {
        var sym = document.getElementById('trade-symbol');
        var name = document.getElementById('trade-name');
        if (window._lastDiagnosisSymbol) { if (sym) sym.value = window._lastDiagnosisSymbol; }
        if (window._lastDiagnosisName) { if (name) name.value = window._lastDiagnosisName; }
        if (!window._lastDiagnosisSymbol && !window._lastDiagnosisName) { if (typeof showToast === 'function') showToast('请先在个股诊断中获取行情'); }
      });
    }

    var fillFromWatchlist = document.getElementById('trade-fill-from-watchlist');
    if (fillFromWatchlist) {
      fillFromWatchlist.addEventListener('click', function () {
        apiGet('/api/market/watchlist')
          .then(function (r) { return r.ok ? r.json() : {}; })
          .then(function (data) {
            var items = data.items || [];
            if (items.length === 0) { if (typeof showToast === 'function') showToast('自选为空'); return; }
            var first = items[0];
            var sym = document.getElementById('trade-symbol');
            var name = document.getElementById('trade-name');
            if (sym) sym.value = first.stock_code || first.code || '';
            if (name) name.value = first.stock_name || first.name || '';
            if (typeof showToast === 'function') showToast('已带入：' + (first.stock_name || first.stock_code));
          })
          .catch(function () { if (typeof showToast === 'function') showToast('请先登录'); });
      });
    }

    if (!accountsEl) return;
    accountsEl.innerHTML = '<p class="text-slate-400 text-sm">加载中…</p>';
    apiGet('/api/trade/dashboard')
      .then(function (r) {
        if (r.status === 401) { accountsEl.innerHTML = '<p class="text-slate-500 text-sm">请先登录</p>'; return null; }
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        if (!data) return;
        var accounts = data.accounts || [];
        var html = '<div class="space-y-2">';
        accounts.forEach(function (a) {
          html += '<div class="flex justify-between text-sm"><span class="text-slate-400">' + ((a.type || a.account_type) === 'sim' ? '模拟盘' : '实盘') + '</span><span class="text-sky-300 font-mono">' + (a.balance != null ? a.balance : '--') + '</span></div>';
        });
        html += '</div>';
        accountsEl.innerHTML = html || '<p class="text-slate-500 text-sm">暂无账户</p>';

        if (tradesEl) {
          var trades = (data.sim_trades || []).slice(0, 10);
          if (trades.length === 0) tradesEl.innerHTML = '<p class="text-slate-500 text-sm">暂无成交</p>';
          else {
            var thtml = '<ul class="space-y-1 text-xs">';
            trades.forEach(function (t) {
              thtml += '<li class="flex justify-between"><span>' + (t.symbol || t.name) + '</span><span class="' + (t.side === 'buy' ? 'text-up' : 'text-down') + '">' + (t.side === 'buy' ? '买' : '卖') + ' ' + (t.open_quantity || '') + '</span></li>';
            });
            thtml += '</ul>';
            tradesEl.innerHTML = thtml;
          }
        }
      })
      .catch(function () {
        accountsEl.innerHTML = '<p class="text-slate-500 text-sm">请先登录</p>';
      });

    var reviewWeekBtn = document.getElementById('trade-ai-review-week');
    var reviewMonthBtn = document.getElementById('trade-ai-review-month');
    var reviewResultEl = document.getElementById('trading-ai-review-result');
    function requestAiReview(period) {
      if (!reviewResultEl) return;
      reviewResultEl.innerHTML = '<span class="text-slate-400">请求中…</span>';
      apiGet('/api/trade/review?period=' + (period === 'month' ? 'month' : 'week'))
        .then(function (r) {
          if (r.status === 401) { reviewResultEl.innerHTML = '<span class="text-slate-500">请先登录</span>'; return null; }
          return r.ok ? r.json() : null;
        })
        .then(function (data) {
          if (data && data.summary) {
            reviewResultEl.innerHTML = '<p class="text-slate-300 whitespace-pre-wrap">' + String(data.summary).replace(/</g, '&lt;').replace(/\n/g, '<br/>') + '</p>';
          } else if (data && data.error) {
            reviewResultEl.innerHTML = '<span class="text-amber-400">' + (data.error || '') + '</span>';
          } else {
            reviewResultEl.innerHTML = '<span class="text-slate-500">暂无该周期成交或接口未返回摘要。</span>';
          }
        })
        .catch(function () {
          reviewResultEl.innerHTML = '<span class="text-slate-500">请先登录；复盘功能将根据近期成交由 AI 生成摘要。</span>';
        });
    }
    if (reviewWeekBtn) reviewWeekBtn.addEventListener('click', function () { requestAiReview('week'); });
    if (reviewMonthBtn) reviewMonthBtn.addEventListener('click', function () { requestAiReview('month'); });
  }

  function submitTrade(formData) {
    apiPost('/api/trade/open', formData)
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.error) throw new Error(j.error);
        if (typeof showToast === 'function') showToast('下单成功');
        loadTrading();
      })
      .catch(function (e) {
        if (typeof showToast === 'function') showToast(e.message || '下单失败');
      });
  }

  // ---------- 社区跟单 ----------
  function loadCommunityFeed() {
    var listEl = document.getElementById('community-feed-list');
    if (!listEl) return;
    listEl.innerHTML = '<p class="text-slate-400 text-sm">加载中…</p>';
    apiGet('/api/market/community/feed')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        var items = data.items || [];
        if (items.length === 0) {
          listEl.innerHTML = '<p class="text-slate-500 text-sm">暂无动态</p>';
          return;
        }
        var sampleHint = (data.source === 'sample') ? '<p class="text-xxs text-slate-500 mb-2">以下为示例动态，非真实用户数据。</p>' : '';
        var html = sampleHint + '<ul class="space-y-4">';
        items.forEach(function (item) {
          html += '<li class="glass p-4 rounded-xl border border-slate-700/50 flex gap-3">';
          html += '<div class="w-10 h-10 rounded-full bg-slate-700 flex-shrink-0"></div>';
          html += '<div class="flex-1 min-w-0"><div class="flex items-center gap-2"><span class="font-medium text-slate-200">' + (item.user || '') + '</span><span class="text-xs ' + (item.action === '买入' ? 'text-up' : 'text-down') + '">' + (item.action || '') + ' ' + (item.symbol || '') + ' @ ' + (item.price ? FormatUtils.formatPrice(item.price) : '--') + '</span></div>';
          html += '<p class="text-sm text-slate-400 mt-1">' + (item.comment || '') + '</p>';
          html += '<div class="text-xxs text-slate-500 mt-1">' + (item.time || '') + '</div></div></li>';
        });
        html += '</ul>';
        listEl.innerHTML = html;
      })
      .catch(function () {
        listEl.innerHTML = '<p class="text-slate-500 text-sm">加载失败</p>';
      });
  }

  // ---------- 系统状态 ----------
  function loadSystemStatus() {
    var el = document.getElementById('system-status-content');
    if (!el) return;
    el.innerHTML = '<p class="text-slate-400 text-sm">加载中…</p>';
    apiGet('/api/system/status')
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        var tip = data.tip ? '<p class="text-xxs text-slate-500 mt-2">' + data.tip + '</p>' : '';
        var docsLink = (data.docs_url ? '<a href="' + data.docs_url + '" target="_blank" rel="noopener" class="text-xxs text-sky-400 hover:underline mt-2 block">API 文档 (开发环境)</a>' : '');
        el.innerHTML = '<div class="space-y-2 text-sm">' +
          '<div class="flex justify-between"><span class="text-slate-400">服务器时间</span><span class="font-mono text-slate-200">' + (data.server_time || '--') + '</span></div>' +
          '<div class="flex justify-between"><span class="text-slate-400">AkShare</span><span class="' + (data.akshare_status === 'OK' ? 'text-green-400' : 'text-amber-400') + '">' + (data.akshare_status || '--') + '</span></div>' +
          '<div class="flex justify-between"><span class="text-slate-400">行情缓存条数</span><span class="font-mono text-slate-200">' + (data.spot_data_rows != null ? data.spot_data_rows : '--') + '</span></div>' +
          '<div class="flex justify-between"><span class="text-slate-400">缓存年龄(秒)</span><span class="font-mono text-slate-200">' + (data.cache_age != null ? data.cache_age : '--') + '</span></div>' +
          tip +
          '<div class="flex gap-2 mt-3">' +
          '<button type="button" id="system-clear-cache-btn" class="flex-1 py-2 rounded text-xs bg-slate-700 hover:bg-slate-600 text-slate-300">清理缓存</button>' +
          '<button type="button" id="system-data-info-btn" class="flex-1 py-2 rounded text-xs bg-slate-700 hover:bg-slate-600 text-slate-300" title="数据来源与说明">数据说明</button>' +
          '</div>' +
          docsLink +
          '</div>';
        var btn = document.getElementById('system-clear-cache-btn');
        if (btn) btn.addEventListener('click', function () {
          btn.disabled = true;
          btn.textContent = '清理中…';
          var h = { 'Content-Type': 'application/json' };
          var token = typeof getToken === 'function' ? getToken() : (window._rox_token || localStorage.getItem('access_token'));
          if (token) h['Authorization'] = 'Bearer ' + token;
          fetch('/api/system/clear-cache', { method: 'POST', headers: h, credentials: 'same-origin' })
            .then(function (r) { return r.json(); })
            .then(function () { btn.textContent = '已清理'; loadSystemStatus(); })
            .catch(function () { btn.disabled = false; btn.textContent = '清理缓存'; });
        });
        var dataInfoBtn = document.getElementById('system-data-info-btn');
        if (dataInfoBtn) dataInfoBtn.addEventListener('click', function () {
          var overlay = document.createElement('div');
          overlay.setAttribute('role', 'dialog');
          overlay.setAttribute('aria-label', '数据说明');
          overlay.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4';
          overlay.innerHTML = '<div class="bg-slate-800 rounded-lg shadow-xl max-w-md w-full p-4 text-sm text-slate-200 space-y-3">' +
            '<div class="flex justify-between items-center"><h3 class="font-medium text-slate-100">数据说明</h3><button type="button" class="text-slate-400 hover:text-white" aria-label="关闭">×</button></div>' +
            '<p class="text-slate-300">行情、指数等数据来自 <strong>AkShare</strong>（东方财富、新浪等公开接口）。北向资金等部分数据来自东方财富网页抓取。数据仅供参考，不构成投资建议。</p>' +
            '<p class="text-xxs text-slate-500">AI 能力需在 .env 中配置 AI_API_KEY 与 AI_BASE_URL，详见项目 docs。</p>' +
            '</div>';
          var close = function () { overlay.remove(); document.removeEventListener('keydown', onKey); };
          overlay.querySelector('button').onclick = close;
          overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
          var onKey = function (e) { if (e.key === 'Escape') { close(); } };
          document.addEventListener('keydown', onKey);
          document.body.appendChild(overlay);
        });
      })
      .catch(function () {
        el.innerHTML = '<p class="text-red-400 text-sm">获取失败</p>';
      });
  }

  function onSwitchMode(mode) {
    if (mode === 'market') loadMarketDashboard();
    // if (mode === 'macro') loadMacro(); // Phase 6 handles this (Charts view)
    if (mode === 'weekly') loadWeekly();
    if (mode === 'screener') loadScreener();
    if (mode === 'diagnosis') initDiagnosisSidebar();
    if (mode === 'kb') initKbSidebar();
    if (mode === 'trading') loadTrading();
    if (mode === 'community') loadCommunityFeed();
    if (mode === 'system') loadSystemStatus();
    if (mode === 'dragon') loadDragonTigerView();
  }

  // ---------- 龙虎榜视图 ----------
  function loadDragonTigerView() {
    var listEl = document.getElementById('dragon-tiger-list');
    var rotationEl = document.getElementById('sector-rotation-list');

    // Load Dragon Tiger data
    if (listEl) {
      listEl.innerHTML = '<p class="text-slate-400 text-sm">加载中...</p>';
      fetch('/api/market/dragon-tiger')
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(function (data) {
          var items = data.data || [];
          if (items.length === 0) {
            listEl.innerHTML = '<p class="text-slate-400 text-sm">暂无龙虎榜数据</p>';
            return;
          }
          listEl.innerHTML = items.slice(0, 20).map(function (item) {
            var changeCls = item.change_pct >= 0 ? 'text-red-400' : 'text-green-400';
            var netCls = (item.net_amount || 0) >= 0 ? 'text-red-400' : 'text-green-400';
            var netDisplay = typeof FormatUtils !== 'undefined' ? FormatUtils.formatBigNumber(item.net_amount || 0) : (item.net_amount || 0);
            return '<div class="flex items-center justify-between py-2 border-b border-slate-700/50 hover:bg-slate-700/30 cursor-pointer" onclick="if(typeof searchStock===\'function\')searchStock(\'' + item.code + '\')">' +
              '<div class="flex-1">' +
              '<span class="text-white text-sm font-medium">' + (item.name || '') + '</span>' +
              '<span class="text-slate-400 text-xs ml-2">' + (item.code || '') + '</span>' +
              '</div>' +
              '<div class="text-right">' +
              '<span class="' + changeCls + ' text-sm">' + (item.change_pct >= 0 ? '+' : '') + (item.change_pct || 0).toFixed(2) + '%</span>' +
              '<span class="text-slate-400 text-xs block">' + (item.reason || '') + '</span>' +
              '<span class="' + netCls + ' text-xs">净买入: ' + netDisplay + '</span>' +
              '</div>' +
              '</div>';
          }).join('');
        })
        .catch(function (err) {
          console.error('龙虎榜加载失败:', err);
          listEl.innerHTML = '<p class="text-red-400 text-sm">加载失败: ' + err.message + '</p>';
        });
    }

    // Load Sector Rotation data
    if (rotationEl) {
      rotationEl.innerHTML = '<p class="text-slate-400 text-sm">加载中...</p>';
      fetch('/api/market/rotation')
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(function (data) {
          var items = data.data || [];
          if (items.length === 0) {
            rotationEl.innerHTML = '<p class="text-slate-400 text-sm">暂无板块轮动数据</p>';
            return;
          }
          rotationEl.innerHTML = items.map(function (item, idx) {
            var pctCls = (item.pct_1d || 0) >= 0 ? 'text-red-400' : 'text-green-400';
            var pct5Cls = (item.pct_5d || 0) >= 0 ? 'text-red-400' : 'text-green-400';
            var rankCls = idx < 3 ? 'bg-red-500/20 text-red-400' : 'bg-slate-600/20 text-slate-400';
            return '<div class="flex items-center justify-between py-2 border-b border-slate-700/50">' +
              '<div class="flex items-center">' +
              '<span class="w-6 h-6 rounded-full ' + rankCls + ' flex items-center justify-center text-xs mr-2">' + (idx + 1) + '</span>' +
              '<span class="text-white text-sm">' + (item.name || '') + '</span>' +
              '</div>' +
              '<div class="text-right">' +
              '<span class="' + pctCls + ' text-sm">' + ((item.pct_1d || 0) >= 0 ? '+' : '') + (item.pct_1d || 0).toFixed(2) + '%</span>' +
              '<span class="' + pct5Cls + ' text-xs block">5日: ' + ((item.pct_5d || 0) >= 0 ? '+' : '') + (item.pct_5d || 0).toFixed(2) + '%</span>' +
              '</div>' +
              '</div>';
          }).join('');
        })
        .catch(function (err) {
          console.error('板块轮动加载失败:', err);
          rotationEl.innerHTML = '<p class="text-red-400 text-sm">加载失败: ' + err.message + '</p>';
        });
    }
  }

  // ---------- 通用图表放大 ----------
  function openLargeChartModal(code, type) {
    var modal = document.getElementById('large-chart-modal');

    // Auto-create modal if missing
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'large-chart-modal';
      modal.className = 'fixed inset-0 z-50 hidden items-center justify-center bg-black/80 backdrop-blur-sm';
      modal.innerHTML = `
            <div class="relative w-[90vw] h-[80vh] bg-slate-900 border border-slate-700 rounded-lg shadow-2xl flex flex-col overflow-hidden">
                <div class="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700">
                    <h3 id="large-chart-title" class="text-lg font-bold text-slate-100">图表详情</h3>
                    <button onclick="closeLargeChartModal()" class="text-slate-400 hover:text-white transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
                <div id="large-chart-body" class="flex-1 w-full h-full p-4 relative"></div>
            </div>
        `;
      document.body.appendChild(modal);
    }

    var body = document.getElementById('large-chart-body');
    var title = document.getElementById('large-chart-title');
    if (!modal || !body) return;

    modal.classList.remove('hidden');
    modal.style.display = 'flex';
    body.innerHTML = '<div id="large-chart-canvas" class="w-full h-full"></div>';

    if (title) title.textContent = (code === '000001' ? '上证指数' : code) + (type === 'fenshi' ? ' 分时图' : ' 详情');

    setTimeout(function () {
      var el = document.getElementById('large-chart-canvas');
      if (!el || typeof echarts === 'undefined') return;

      var ch = echarts.init(el);
      ch.showLoading({ color: '#38bdf8', textColor: '#94a3b8', maskColor: 'rgba(15, 23, 42, 0.6)' });

      if (type === 'fenshi') {
        apiGet('/api/market/fenshi?code=' + code)
          .then(function (r) { return r.ok ? r.json() : {}; })
          .then(function (d) {
            ch.hideLoading();
            var times = d.times || [];
            var prices = d.prices || [];
            var volumes = d.volumes || [];
            if (!times.length) { el.innerHTML = '<div class="flex h-full items-center justify-center text-slate-500">暂无数据</div>'; return; }

            ch.setOption({
              backgroundColor: 'transparent',
              tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                backgroundColor: 'rgba(30,41,59,0.95)',
                borderColor: '#334155',
                textStyle: { color: '#e2e8f0' },
                formatter: function (params) {
                  if (!params || !params.length) return '';
                  var idx = params[0].dataIndex;
                  var t = times[idx] || '';
                  var p = prices[idx] != null ? prices[idx].toFixed(2) : '--';
                  var v = volumes[idx] != null ? (volumes[idx] >= 10000 ? (volumes[idx] / 10000).toFixed(1) + '万' : volumes[idx]) : '--';
                  return t + '<br/>价格: ' + p + '<br/>成交量: ' + v;
                }
              },
              grid: [{ left: 60, right: 60, top: 30, bottom: 80 }, { left: 60, right: 60, top: '85%', height: '10%' }],
              xAxis: [
                { type: 'category', data: times, gridIndex: 0, axisLabel: { color: '#94a3b8' }, boundaryGap: false },
                { type: 'category', data: times, gridIndex: 1, axisLabel: { show: false } }
              ],
              yAxis: [
                { type: 'value', gridIndex: 0, scale: true, splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
                { type: 'value', gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }
              ],
              dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], bottom: 10 }],
              series: [
                { name: '价格', type: 'line', data: prices, smooth: true, symbol: 'none', lineStyle: { color: '#38bdf8', width: 2 }, areaStyle: { color: 'rgba(56,189,248,0.2)' } },
                { name: '成交量', type: 'bar', data: volumes, xAxisIndex: 1, yAxisIndex: 1, itemStyle: { color: function (p) { return (prices[p.dataIndex] >= (prices[p.dataIndex - 1] || prices[p.dataIndex])) ? '#ef4444' : '#22c55e'; } } }
              ]
            });
          })
          .catch(function () { ch.hideLoading(); el.innerHTML = '<div class="flex h-full items-center justify-center text-slate-500">加载失败</div>'; });
      } else if (type === 'prediction') {
        apiGet('/api/market/prediction?horizon=5')
          .then(function (r) { return r.ok ? r.json() : {}; })
          .then(function (data) {
            ch.hideLoading();
            var history = data.history || {};
            var histDates = history.dates || [];
            var histPrices = history.prices || [];
            var predDates = data.dates || [];
            var predPrices = data.prices || [];

            if (!histDates.length) { el.innerHTML = '<div class="flex h-full items-center justify-center text-slate-500">暂无数据</div>'; return; }

            // Stitch data
            var fullDates = histDates.concat(predDates);
            var fullHist = histPrices.concat(new Array(predDates.length).fill(null));
            // For prediction line, we need to connect the last history point
            var lastHistPrice = histPrices[histPrices.length - 1];
            var fullPred = new Array(histPrices.length - 1).fill(null);
            fullPred.push(lastHistPrice);
            fullPred = fullPred.concat(predPrices);

            ch.setOption({
              backgroundColor: 'transparent',
              tooltip: { trigger: 'axis', backgroundColor: 'rgba(30,41,59,0.95)', borderColor: '#334155', textStyle: { color: '#e2e8f0' } },
              grid: { left: 50, right: 30, top: 30, bottom: 30, containLabel: true },
              xAxis: { type: 'category', data: fullDates, axisLabel: { color: '#94a3b8' } },
              yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
              series: [
                { name: '历史', type: 'line', data: fullHist, smooth: true, lineStyle: { color: '#38bdf8', width: 2 } },
                { name: '预测', type: 'line', data: fullPred, smooth: true, lineStyle: { color: '#facc15', type: 'dashed', width: 2 } }
              ]
            });
          })
          .catch(function () { ch.hideLoading(); el.innerHTML = '<div class="flex h-full items-center justify-center text-slate-500">加载失败</div>'; });
      }

      // Resize handler
      window.addEventListener('resize', function () { ch.resize(); });
      window._largeChartInstance = ch;
    }, 100);
  }

  function closeLargeChartModal() {
    var modal = document.getElementById('large-chart-modal');
    if (modal) {
      modal.classList.add('hidden');
      modal.style.display = 'none';
    }
    if (window._largeChartInstance) {
      window._largeChartInstance.dispose();
      window._largeChartInstance = null;
    }
  }
  window.openLargeChartModal = openLargeChartModal;
  window.closeLargeChartModal = closeLargeChartModal;

  window.rox1Views = {
    loadMarketDashboard: loadMarketDashboard,
    loadMacro: loadMacro,
    loadWeekly: loadWeekly,
    loadScreener: loadScreener,
    initDiagnosisSidebar: initDiagnosisSidebar,
    runDiagnosis: runDiagnosis,
    loadKbSearch: loadKbSearch,
    loadTrading: loadTrading,
    submitTrade: submitTrade,
    loadCommunityFeed: loadCommunityFeed,
    loadSystemStatus: loadSystemStatus,
    onSwitchMode: onSwitchMode
  };
  // ---------- 交易模块 ----------
  function loadTrading() {
    var accountSelect = document.getElementById('trade-account');
    if (accountSelect) {
      // Re-bind change event if not bound
      if (!accountSelect._bound) {
        accountSelect._bound = true;
        accountSelect.onchange = function () { loadTradingData(); };
      }
    }
    loadTradingData();
  }

  function loadTradingData() {
    var accEl = document.getElementById('trading-accounts');
    var listEl = document.getElementById('trading-trades-list');
    var mode = 'sim';
    var accountSelect = document.getElementById('trade-account');
    if (accountSelect) mode = accountSelect.value;

    if (accEl) accEl.innerHTML = '<p class="text-slate-400 text-sm">加载中...</p>';

    // 1. Account Summary
    apiGet('/api/portfolio/summary?mode=' + mode)
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (data) {
        if (!accEl) return;
        if (data.message && mode === 'real') {
          accEl.innerHTML = '<div class="p-4 bg-amber-500/10 border border-amber-500/30 rounded text-amber-300 text-sm">' +
            '<i class="fas fa-exclamation-triangle mr-2"></i>' + data.message +
            '</div>';
          return;
        }
        var bal = FormatUtils.formatBigNumber(data.balance || 0);
        var asset = FormatUtils.formatBigNumber(data.total_asset || data.balance || 0);
        var pnl = (data.total_pnl || 0);
        var pnlCls = FormatUtils.getColorClass(pnl);
        accEl.innerHTML = '<div class="grid grid-cols-2 gap-4 text-center">' +
          '<div class="bg-slate-900 rounded p-3"><div class="text-xs text-slate-500">总资产</div><div class="text-lg font-mono text-slate-200">' + asset + '</div></div>' +
          '<div class="bg-slate-900 rounded p-3"><div class="text-xs text-slate-500">可用资金</div><div class="text-lg font-mono text-slate-200">' + bal + '</div></div>' +
          '<div class="col-span-2 bg-slate-900 rounded p-3"><div class="text-xs text-slate-500">参考盈亏</div><div class="text-lg font-mono ' + pnlCls + '">' + (pnl >= 0 ? '+' : '') + FormatUtils.formatBigNumber(pnl) + '</div></div>' +
          '</div>';
      })
      .catch(function () { if (accEl) accEl.innerHTML = '<p class="text-slate-500 text-sm">加载失败</p>'; });

    // 2. Positions (Optional display)
    // apiGet('/api/portfolio/positions?mode=' + mode)...
  }

  // Trading Form Handling
  var tradeForm = document.getElementById('trading-form');
  if (tradeForm) {
    tradeForm.onsubmit = function (e) {
      e.preventDefault();
      var btn = document.getElementById('trading-submit-btn');
      if (btn) btn.disabled = true;

      var mode = 'sim';
      var accountSelect = document.getElementById('trade-account');
      if (accountSelect) mode = accountSelect.value;

      var sym = document.getElementById('trade-symbol').value;
      var side = document.getElementById('trade-side').value;
      var price = parseFloat(document.getElementById('trade-price').value);
      var qty = parseInt(document.getElementById('trade-qty').value);
      var name = document.getElementById('trade-name').value;

      apiPost('/api/portfolio/order', {
        symbol: sym,
        side: side,
        price: price,
        quantity: qty,
        name: name,
        mode: mode
      }).then(function (r) {
        if (r.ok) return r.json();
        return r.json().then(function (d) { throw new Error(d.detail || '下单失败'); });
      }).then(function (d) {
        if (typeof showToast === 'function') showToast("✅ 下单成功: " + (d.message || '已提交'));
        if (btn) btn.disabled = false;
        loadTradingData(); // Refresh assets
      }).catch(function (e) {
        if (typeof showToast === 'function') showToast("❌ 下单失败: " + e.message);
        if (btn) btn.disabled = false;
      });
    };
  }

  window.onSwitchMode = onSwitchMode;
})();
