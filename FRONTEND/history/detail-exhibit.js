(() => {
  const events = [
    { file:'event-11th-century.html', key:'event-seal', scene:['洞窟与小室','封藏的材料','仍待辨析的原因'], image:'../assets/history-generated/01-文物封藏.png' },
    { file:'event-1900.html', key:'event-1900', scene:['发现被封住的门','报告与保管','发现之后'], image:'../assets/history-generated/02-王圆箓发现藏经洞.png' },
    { file:'event-1907.html', key:'event-1907', scene:['抵达敦煌','材料的去向','分散后的研究'], image:'../assets/history-generated/03-斯坦因来访.png' },
    { file:'event-1908.html', key:'event-pelliot', scene:['在烛光下翻阅','选择与收藏','知识史的后果'], image:'../assets/history-generated/04-伯希和入窟.png' },
    { file:'event-1912-1924.html', key:'event-dispersal', scene:['不止一个事件','考察与收藏网络','记录留下的边界'], image:'../assets/history-generated/05-多国考察.png' },
    { file:'event-1944.html', key:'event-protection', scene:['机构化的起点','记录成为保护','延续的基础'], image:'../assets/history-generated/06-常书鸿保护.png' },
    { file:'event-contemporary.html', key:'event-contemporary', scene:['1984 · 研究体系建立','1994 · International Dunhuang Programme','数字时代 · 重新建立关系'], image:'../assets/history-generated/07-数字重联.png' }
  ];
  const file = window.location.pathname.split('/').pop();
  const currentIndex = events.findIndex(event => event.file === file);
  if (currentIndex < 0) return;
  const event = events[currentIndex];
  const content = {
    'event-seal': { year:'11世纪初', title:'文物封藏', summary:'大量佛教经卷、社会文书、绘画、纺织品、法器等被放入洞窟，洞口随后封闭。文献同时指出，具体日期与封藏原因仍存在争议。', paragraphs:['莫高窟第 17 窟，即后来所称的“藏经洞”，位于第 16 窟甬道内。IDP 的时间线指出，这个小室最初建于 9 世纪，与当地僧人洪辩的纪念活动有关；它并非一开始就被设计成今日所见的“图书馆”。','洞内后来汇集了数量极大的写本、印刷品、绘画及其他物件，材料年代大体跨越 4 世纪至 11 世纪初。现有资料通常将封闭时间放在 11 世纪初，这一封存使不同语言、宗教和日常书写材料得以在相对封闭的环境中留存数百年。','关于封闭的具体原因，学界仍有不同解释，不能简单归结为某一次确定的战乱或单一命令。正是这种尚待解释的封闭过程，使第 17 窟既是敦煌文献保存史的关键节点，也提醒研究者必须区分可靠证据与后来的推测。'], sources:[['International Dunhuang Programme · History of the IDP','https://idp.bl.uk/about/history-of-the-idp/'],['International Dunhuang Programme · A timeline of Dunhuang','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/introducing-dunhuang/a-timeline-of-dunhuang/']] },
    'event-1900': { year:'1900年', title:'王圆箓发现藏经洞', summary:'王圆箓在清理莫高窟第 16 窟甬道积尘时发现隐藏洞室，随后向地方官员报告，并卷入文书转移与售出的复杂过程。', paragraphs:['IDP 资料记载，王圆箓在 19 世纪末来到莫高窟并参与修缮。1900 年，他在第 16 窟甬道发现一扇被封住的门，门后的小室后来被确认为第 17 窟。洞中密集保存的写本、绘画和印刷品，使这一发现成为近代敦煌史的转折点。','发现之后，王圆箓向地方官员报告，也曾以部分材料争取修缮经费；他随后成为洞藏物件的实际保管者和钥匙持有者之一。材料的发现并没有立即进入一套完善的文物保护程序，地方资源、行政响应和当时的制度条件都深刻影响了之后的处置。','因此，理解王圆箓不能只套用单一的道德标签。他的行动发生在清末边地、经费匮乏与近代文物制度尚未成熟的背景中；同时，洞藏后续向多国收藏分散的过程，也确实构成今日研究和保护必须正视的历史问题。'], sources:[['International Dunhuang Programme · A timeline of Dunhuang','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/introducing-dunhuang/a-timeline-of-dunhuang/'],['International Dunhuang Programme · Discovering Cave 17','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/cave-17-the-library-cave/discovering-cave-17/']] },
    'event-1907': { year:'1907年', title:'斯坦因来到敦煌', summary:'斯坦因通过王圆箓和翻译蒋孝琬获得大量文物，相关收藏主要流入英国和印度的博物馆、图书馆，成为藏经洞文物流散的重要转折。', paragraphs:['1907 年，奥雷尔·斯坦因在第二次中亚考察途中抵达敦煌。他原本关注汉代边塞遗址，到达后得知第 17 窟的发现，并在王圆箓允许下接触洞内材料。IDP 的记录显示，他的考察与藏经洞文献进入国际收藏体系密切相关。','斯坦因取得的材料包括大量写本，以及绘画、纺织品、印刷品和少量其他物件。它们随后主要进入英国的博物馆、图书馆与维多利亚和阿尔伯特博物馆，以及印度新德里的相关收藏。现存目录和影像资料也让这些材料成为敦煌学研究的重要基础。','这次获取同时是敦煌文物流散史的重要节点：材料的跨国保存扩展了研究视野，却也使原本同处一室的文献分散于不同机构。今天的整理、数字化与跨馆目录工作，正是在承认这段历史的基础上重新建立材料之间可核验的联系。'], sources:[['International Dunhuang Programme · A timeline of Dunhuang','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/introducing-dunhuang/a-timeline-of-dunhuang/'],['International Dunhuang Programme · History of the IDP','https://idp.bl.uk/about/history-of-the-idp/']] },
    'event-pelliot': { year:'1908年', title:'伯希和入窟', summary:'伯希和凭借语言能力在烛光下细看文书，挑选数千件；他在北京展示部分文物，也推动余卷运往北京。', paragraphs:['法国学者伯希和在 1906—1909 年的中亚考察期间停留敦煌，并于 1908 年进入藏经洞现场翻阅文献。IDP 时间线记载，他在敦煌停留数月，取得了大量写本、便携绘画、纺织品和其他物件；其语言能力使他能够在材料选择中进行细读与辨识。','这些材料后来主要由法国国家图书馆和吉美国立亚洲艺术博物馆保存。法国国家图书馆的相关资料也确认，伯希和于 1908 年在敦煌取得的手稿构成其敦煌收藏的重要来源。由此形成的“伯希和系列”至今仍是研究汉文、藏文及多语种敦煌文献的重要资源。','这段经历不应被简化为一段传奇式的个人故事。它既使大量材料获得长期保存、编目和研究，也加深了藏经洞文献在全球机构间的分散。理解伯希和的选择，必须同时看到知识史上的影响与文物流散带来的结构性后果。'], sources:[['International Dunhuang Programme · A timeline of Dunhuang','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/introducing-dunhuang/a-timeline-of-dunhuang/'],['Bibliothèque nationale de France · Chronique n°49','https://multimedia-ext.bnf.fr/Chroniques/chroniques_49.pdf']] },
    'event-dispersal': { year:'1912—1924年', title:'多国考察者相继到来', summary:'大谷探险队、鄂登堡俄罗斯考察团、朗登·华纳等先后到达敦煌，带走或记录文书、绘画、壁画碎片和雕塑；这些记录具有价值，但流散与移除也造成了损失。', paragraphs:['这一阶段并不是单一事件。1910 年，中国教育部门下令将莫高窟余下的汉文写本运往北京，它们后来成为国家图书馆敦煌收藏的重要基础。同一时期，来自日本、俄国和美国的考察活动也以不同目的进入敦煌及其周边地区。','IDP 记载，1910—1914 年大谷探险队在敦煌获得数百件写本，现分藏于中国、日本和韩国等地；1914—1915 年，俄国奥登堡的考察对莫高窟进行了系统调查，并有材料进入圣彼得堡收藏。它们都属于近代考察、收藏与研究网络的一部分。','1924 年，朗登·华纳抵达莫高窟，哈佛图书馆保存的资料强调他拍摄了大量壁画与彩塑影像。讨论这一时期时需要区分对象和行为：华纳的敦煌活动主要关联洞窟艺术与摄影，不能笼统表述为从藏经洞带走大量经卷。不同机构留下的记录和收藏，共同构成这段复杂的传播与流散史。'], sources:[['International Dunhuang Programme · A timeline of Dunhuang','https://idp.bl.uk/discover/learning/dunhuang/dunhuang-articles/introducing-dunhuang/a-timeline-of-dunhuang/'],['International Dunhuang Programme · National Library of China Collection','https://idp.bl.uk/discover/collection-categories/the-national-library-of-china-collection/'],['Harvard Library · Langdon Warner Photographs','https://library.harvard.edu/exhibits/langdon-warner-photographs-1924-dunhuang-expedition']] },
    'event-protection': { year:'1944年', title:'常书鸿开启专业保护', summary:'国家敦煌艺术研究所成立，开展系统保护、调查、临摹、编号与记录工作，成为今天敦煌研究院的前身。', paragraphs:['1944 年 1 月 1 日，国立敦煌艺术研究所正式成立，隶属当时教育部，常书鸿任所长。敦煌研究院的历史资料将这一时刻视为制度化保护与研究工作的重要起点：莫高窟不再只是被动等待修缮的遗址，而开始拥有相对稳定的专业机构。','研究所的工作不仅是看守洞窟，也包括调查、测绘、临摹、摄影、资料整理与学术研究。对于石窟这样由壁画、彩塑、建筑与环境共同构成的文化遗产，持续记录和专业人员的长期在场，是后续保护决策能够被复核的重要条件。','机构沿革也显示出这种工作的连续性：1950 年更名为敦煌文物研究所，1984 年扩建为敦煌研究院。名称和组织形式虽有变化，但保护、研究与资料积累逐步形成了延续至今的基础。常书鸿与同事们的工作因此应被理解为现代敦煌保护体系的一段开端。'], sources:[['敦煌研究院 · 国立敦煌艺术研究所时期','https://www.dha.ac.cn/info/1006/2239.htm'],['敦煌研究院 · 机构简介','https://www.dha.ac.cn/jggk/jgjj.htm']] },
    'event-contemporary': { year:'20世纪至今', title:'敦煌学与数字重联', summary:'敦煌学逐渐成为国际跨学科领域。摄影、数字化、数据库与国际合作，让分散在世界各地的文物和信息拥有重新建立关联的可能。', paragraphs:['20 世纪后半叶，敦煌保护与研究逐步形成更稳定的机构体系：敦煌研究院资料显示，1984 年敦煌文物研究所扩建为敦煌研究院。与此同时，散藏在世界各地的敦煌文献使研究者长期面对目录不统一、图像难获取和材料难以比对的问题。','International Dunhuang Programme 于 1994 年在英国图书馆建立，其重要动因之一正是藏经洞材料的国际分散。IDP 在 1995 年推出早期网站，随后通过数字图像、联合目录与可同步的数据库，让不同收藏机构能够保留各自数据权利的同时建立研究层面的连接。','“数字重联”并不等于实物已全部回归原处。它更接近一种可检索、可比对、可追溯的工作方法：把同一题记、同一卷轴的碎片、同一收藏史或相关图像重新置入可核验的关系网中。数字化因此服务于研究与保护，而不能替代对实物、出处和机构责任的讨论。'], sources:[['International Dunhuang Programme · History of the IDP','https://idp.bl.uk/about/history-of-the-idp/'],['敦煌研究院 · 机构简介','https://www.dha.ac.cn/jggk/jgjj.htm']] }
  };
  Object.assign(event, content[event.key]);
  const shell = document.querySelector('main[data-history-event]');
  if (shell) {
    shell.className = 'page';
    shell.innerHTML = `<section class="page-hero"><div class="eyebrow">${event.year} · 历史场景示意图</div><h1>${event.title}</h1><p>${event.summary}</p></section><img class="detail-image" src="${event.image}" alt="${event.title}历史场景示意图"><section class="history-detail-copy" aria-label="历史介绍">${event.paragraphs.map(paragraph => `<p>${paragraph}</p>`).join('')}</section><footer class="history-sources"><span>资料来源</span>${event.sources.map(source => `<a href="${source[1]}" target="_blank" rel="noreferrer">${source[0]}</a>`).join('')}</footer><p class="footer"><a href="index.html">← 返回时间线</a></p>`;
  }
  const previous = events[currentIndex - 1];
  const next = events[currentIndex + 1];
  const returnUrl = new URL('index.html', window.location.href);
  returnUrl.searchParams.set('chapter', event.key);
  const main = document.querySelector('main.page');
  const hero = main?.querySelector('.page-hero');
  const image = main?.querySelector('.detail-image');
  const copy = main?.querySelector('.history-detail-copy');
  const sources = main?.querySelector('.history-sources');
  const originalReturn = main?.querySelector('.footer a');
  if (!main || !hero || !image || !copy || !sources || !originalReturn) return;

  document.body.classList.add('history-exhibit', 'js-exhibit');
  hero.classList.add('event-cover');
  image.classList.add('event-cover-image');
  hero.prepend(image);
  const backToTimeline = document.createElement('a');
  backToTimeline.className = 'event-back-link';
  backToTimeline.href = `index.html?${returnUrl.searchParams}`;
  backToTimeline.textContent = '← 返回时间线';
  hero.append(backToTimeline);
  const continueReading = document.createElement('a');
  continueReading.className = 'event-continue exhibit-reveal';
  continueReading.href = '#narrative';
  continueReading.textContent = '继续阅读 ↓';
  hero.append(continueReading);
  hero.querySelector('h1')?.classList.add('exhibit-reveal');
  hero.querySelector('p')?.classList.add('exhibit-reveal');

  const paragraphs = [...copy.querySelectorAll('p')];
  copy.id = 'narrative';
  copy.className = 'exhibit-narrative';
  copy.innerHTML = '';
  const narrativeImage = image.cloneNode(true);
  narrativeImage.alt = '';
  const media = document.createElement('div');
  media.className = 'narrative-media exhibit-reveal';
  media.append(narrativeImage);
  const scenes = document.createElement('div');
  scenes.className = 'narrative-scenes';
  paragraphs.forEach((paragraph, index) => {
    const scene = document.createElement('article');
    scene.className = 'narrative-scene exhibit-reveal';
    scene.innerHTML = `<span class="scene-index">0${index + 1}</span><h2>${event.scene[index] || '阅读这一段历史'}</h2>`;
    scene.append(paragraph);
    scenes.append(scene);
  });
  copy.append(media, scenes);

  const sourceLinks = [...sources.querySelectorAll('a')];
  sources.className = 'exhibit-sources exhibit-reveal';
  sources.innerHTML = `<div class="eyebrow">证据与出处</div><h2>资料来源</h2><div class="source-list">${sourceLinks.map(link => `<a href="${link.href}" target="_blank" rel="noreferrer"><small>外部资料</small><b>${link.textContent}</b><i aria-hidden="true">→</i></a>`).join('')}</div>`;

  const chapterNavigation = document.createElement('nav');
  chapterNavigation.className = 'chapter-navigation exhibit-reveal';
  const chapterCard = (entry, direction) => entry ? `<a class="chapter-card" href="${entry.file}"><span><small>${direction}</small><strong>${entry.key === 'event-seal' ? '文物封藏' : entry.key === 'event-1900' ? '王圆箓发现藏经洞' : entry.key === 'event-1907' ? '斯坦因来到敦煌' : entry.key === 'event-pelliot' ? '伯希和入窟' : entry.key === 'event-dispersal' ? '多国考察者相继到来' : entry.key === 'event-protection' ? '常书鸿开启专业保护' : '敦煌学与数字重联'}</strong></span></a>` : '<div></div>';
  chapterNavigation.innerHTML = `${chapterCard(previous, '← 上一章')}<a class="return-timeline" href="${returnUrl.pathname.split('/').pop()}?${returnUrl.searchParams}">返回历史总览</a>${chapterCard(next, '下一章 →')}`;
  originalReturn.parentElement.replaceWith(chapterNavigation);

  const revealTargets = [...document.querySelectorAll('.exhibit-reveal')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reducedMotion || !('IntersectionObserver' in window)) revealTargets.forEach(target => target.classList.add('is-revealed'));
  else {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('is-revealed'); observer.unobserve(entry.target); } }), { threshold:.14, rootMargin:'0px 0px -7%' });
    revealTargets.forEach(target => observer.observe(target));
  }
})();
